"""BiTraP-NP with losses restricted to the real future steps.

Subclasses the public ``BiTraPNP``; on complete futures it reduces to upstream.
"""

import sys

import torch
import torch.nn.functional as F
from torch.nn.utils import rnn


def import_upstream(repo):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from bitrap.modeling.bitrap_np import BiTraPNP
    from configs.defaults import _C

    return BiTraPNP, _C.clone()


def model_cfg(defaults):
    """BiTraP's JAAD settings for a 15 -> 45 task."""
    cfg = defaults.MODEL.clone()
    cfg.defrost()
    cfg.merge_from_list([
        "INPUT_LEN", 15, "PRED_LEN", 45, "BEST_OF_MANY", True, "K", 20,
        "LATENT_DIST", "gaussian", "LATENT_DIM", 32,
        "GLOBAL_EMBED_SIZE", 256, "GLOBAL_HIDDEN_SIZE", 256, "GLOBAL_INPUT_DIM", 4,
        "DROPOUT", 0.0, "WITH_FLOW", False, "INPUT_EMBED_SIZE", 256,
        "ENC_HIDDEN_SIZE", 256, "DEC_INPUT_SIZE", 256, "DEC_HIDDEN_SIZE", 256,
        "DEC_OUTPUT_DIM", 4, "DEC_WITH_Z", False,
    ])
    cfg.freeze()
    return cfg


def masked_cvae_loss(pred_goal, pred_traj, target, mask, best_of_many=True):
    """pred_goal (B, K, 4); pred_traj (B, T, K, 4); target (B, T, 4); mask (B, T)."""
    K = pred_goal.shape[1]
    tgt = target.unsqueeze(2).expand(-1, -1, K, -1)
    step_err = torch.sqrt(((pred_traj - tgt) ** 2).sum(-1))
    traj_rmse = (step_err * mask.to(pred_traj.dtype).unsqueeze(-1)).sum(1)
    goal_rmse = torch.sqrt(((pred_goal - tgt[:, -1]) ** 2).sum(-1))
    full = mask[:, -1]
    if best_of_many:
        best = torch.where(full, goal_rmse.argmin(1), traj_rmse.argmin(1))
        rows = torch.arange(len(best), device=best.device)
        g, t = goal_rmse[rows, best], traj_rmse[rows, best]
    else:
        g, t = goal_rmse.mean(1), traj_rmse.mean(1)
    loss_goal = g[full].mean() if full.any() else goal_rmse.sum() * 0.0
    return loss_goal, t.mean()


def build(repo, device):
    BiTraPNP, defaults = import_upstream(repo)

    class MaskedBiTraPNP(BiTraPNP):
        def gaussian_latent_net(self, enc_h, cur_state, target=None, lengths=None):
            p = self.p_z_x(enc_h)
            mu_p, logvar_p = p[:, :self.cfg.LATENT_DIM], p[:, self.cfg.LATENT_DIM:]
            if target is not None:
                h0 = self.node_future_encoder_h(cur_state)
                h0 = torch.stack([h0, torch.zeros_like(h0)], dim=0)
                # the posterior encoder only sees the real part of the future
                packed = rnn.pack_padded_sequence(target, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, target_h = self.gt_goal_encoder(packed, h0)
                target_h = target_h.permute(1, 0, 2).reshape(enc_h.shape[0], -1)
                target_h = F.dropout(target_h, p=0.25, training=self.training)
                q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
                mu, logvar = q[:, :self.cfg.LATENT_DIM], q[:, self.cfg.LATENT_DIM:]
                kld = 0.5 * ((logvar.exp() / logvar_p.exp())
                             + (mu_p - mu).pow(2) / logvar_p.exp() - 1 + (logvar_p - logvar))
                kld = torch.clamp(kld.sum(-1).mean(), min=0.001)
            else:
                mu, logvar, kld = mu_p, logvar_p, 0.0
            eps = torch.randn(enc_h.shape[0], self.K, self.cfg.LATENT_DIM, device=enc_h.device)
            return mu.unsqueeze(1) + eps * torch.exp(0.5 * logvar).unsqueeze(1), kld

        def forward(self, input_x, target_y=None, target_mask=None):
            """Returns (pred_goal, pred_traj (B, T, K, 4), losses)."""
            cur_pos = input_x[:, -1, :]
            h_x = self.encoder(input_x)
            lengths = target_mask.sum(1) if target_y is not None else None
            Z, kld = self.gaussian_latent_net(h_x, cur_pos, target_y, lengths)
            enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
            pred_goal = self.goal_decoder(enc_h_and_z)
            pred_traj = self.pred_future_traj(enc_h_and_z if self.cfg.DEC_WITH_Z else h_x, pred_goal)
            pred_goal = pred_goal + cur_pos.unsqueeze(1)
            pred_traj = pred_traj + cur_pos[:, None, None, :]
            loss = {}
            if target_y is not None:
                loss["loss_goal"], loss["loss_traj"] = masked_cvae_loss(
                    pred_goal, pred_traj, target_y, target_mask, self.cfg.BEST_OF_MANY)
                loss["loss_kld"] = kld
            return pred_goal, pred_traj, loss

    return MaskedBiTraPNP(model_cfg(defaults)).to(device)


def kld_scheduler(device):
    from bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal

    ps = ParamScheduler()
    ps.create_new_scheduler(name="kld_weight", annealer=sigmoid_anneal, annealer_kws={
        "device": device, "start": 0, "finish": 100.0, "center_step": 400.0, "steps_lo_to_hi": 100.0})
    return ps
