# MotoVerse — A Benchmark for Predictive Perception of Motorized Two-Wheelers in Dense Unstructured Traffic

Code release accompanying the WACV 2027 paper *MotoVerse: A Benchmark for Predictive Perception of Motorized Two-Wheelers in Dense Unstructured Traffic*.

## Abstract

Motorized Two-Wheelers (MTWs) dominate personal transport across much of the Global South, yet remain largely overlooked in predictive perception research for autonomous driving and ADAS. We introduce MotoVerse, a benchmark for predictive perception from an MTW's ego-centric perspective. MotoVerse comprises 949 maneuver clips collected in dense, unstructured traffic and trajectory annotations for over 31,000 surrounding road agents, including over 16,000 MTWs, enabling two complementary tasks: Ego-Rider Maneuver Anticipation (ERMA) and Traffic Agent Trajectory Prediction (TATP). Each clip is further annotated with an MTW-specific challenge taxonomy comprising dynamic ego-motion, weak traffic structure and contextual occlusion, enabling challenge-aware evaluation. We benchmark established conventional prediction models, general-purpose Vision-Language Models (VLMs), driving-specific VLMs and vision-language-action models. Our experiments reveal three key insights: (i) video backbones reach most of their full-observation accuracy from the first fifth of a maneuver, whereas zero-shot VLMs need most of the maneuver to be observed; (ii) zero-shot VLMs, whether general-purpose or driving-specific, largely remain close to simple baselines on both tasks; a preliminary LoRA study shows that adaptation lifts anticipation sharply but leaves trajectory prediction behind conventional models; and (iii) the rider's own motion drives forecasting difficulty: for every model, including a static baseline, errors are highest during left turns and rise under dynamic ego-motion. We believe MotoVerse will serve as a challenging benchmark for safer and more robust predictive AI for intelligent two-wheelers.

## Setup

```bash
conda create -n motoverse python=3.11 -y
conda activate motoverse
pip install -r requirements.txt

git clone https://github.com/facebookresearch/SlowFast third_party/SlowFast
git -C third_party/SlowFast checkout 88bb4c9
git -C third_party/SlowFast apply ../../patches/slowfast.patch
git clone https://github.com/umautobots/bidireaction-trajectory-prediction third_party/bidireaction-trajectory-prediction
```

The VLMs need their own environments: `requirements-vlm.txt` (Qwen3-VL, Cosmos-Reason2, UniDriveVLA, Alpamayo, LoRA) and `requirements-internvl.txt` (InternVL2.5, DriveLMM-o1).

## Data

Place the dataset under `data/MotoVerse` (paths are set in `configs/paths.yaml`):

```
data/MotoVerse/
  clips/<clip_id>.mp4
  erma.csv    clip_id, session, split, maneuver, man_start_s, man_end_s, wla, co, dem
  tatp.csv    clip_id, track_id, agent_type, split, frame, x1, y1, x2, y2   (960x540 pixels)
```

```bash
python tools/prepare_erma.py      # tau-truncated 256x256 ERMA clips
python tools/extract_frames.py    # 960x540 TATP frames (needed for the VLMs only)
```

## ERMA

Kinetics-400 initialisations go in `outputs/pretrained`: [I3D](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl), [SlowFast](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl), [X3D-M](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth) and [C2D](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D_8x8_R50.pyth), followed by `python tools/convert_c2d.py`.

```bash
for m in C2D I3D SlowFast X3D; do for t in 0.2 0.3 0.5 1.0; do
  python train_erma.py --model $m --tau $t
done; done
python eval_erma.py
```

Each backbone is trained separately at every observation ratio tau; the checkpoint with the best validation accuracy is scored on test.

## TATP

```bash
python train_tatp.py --model PIETraj
python train_tatp.py --model BiTraP
python eval_tatp.py
```

Models observe 15 frames and predict the next 45. Metrics are computed in 960x540 pixels on test windows with the full 1.5 s future, alongside a static baseline that holds the last observed box.

## VLMs

```bash
python tools/prepare_vlms.py                              # DriveLMM-o1 and Alpamayo-1.5-10B
python run_vlm.py --model Qwen3-VL-4B --task erma
python run_vlm.py --model Qwen3-VL-4B --task tatp
python score_vlm.py --task erma
python score_vlm.py --task tatp
```

ERMA accuracy is computed over parsed answers and reported with the parse rate. TATP is evaluated on one window per test track run (1,246 windows); an unparsed answer holds the last observed box.

LoRA adaptation of Qwen3-VL-2B and UniDriveVLA:

```bash
python train_lora.py --model Qwen3-VL-2B --task erma --tau 0.5
python run_vlm.py --model Qwen3-VL-2B --task erma --taus 0.5 --adapter outputs/lora/Qwen3-VL-2B_erma_tau0.5/final --name Qwen3-VL-2B-LoRA
```
