# Fantastic-Four: Ghost Vessel Detector

## Summary

Ghost Fleet Trigger is an onboard maritime-security pipeline that converts xView3 Sentinel-1 SAR imagery into compact dark-vessel alerts. The system uses TerraMind-1.0-small to score vessel candidates from SAR crops, estimates fishing probability and vessel length, then cross-matches detections against a local AIS cache. A detection with no nearby AIS signature is emitted as a high-value law-enforcement alert instead of downlinking the full SAR scene.

Target users include coast guards, sanctions-enforcement teams, illegal-fishing monitors, and commodity-risk analysts.

## The Problem

AIS is a cooperative broadcast system: vessels announce their position when the transponder is on and honest. Illegal fishing vessels, sanctions evasions, and ship-to-ship transfers can avoid normal monitoring by disabling or spoofing AIS. SAR imagery provides an independent observation channel because radar can detect vessels at night and through cloud cover.

The objective is to turn large SAR scenes into actionable messages:

```text
Vessel detected at (lat, lon) with no matching AIS signature.
Classification: likely vessel / likely fishing vessel.
Confidence: model score.
Estimated length: meters.
```

## Technical Approach

The pipeline has four stages:

1. Candidate proposal: SAR brightness, local contrast, cross-polarization support, and small-object band-pass response identify likely ship locations in a downsampled scene overview.
2. TerraMind scoring: each candidate is cropped from VV/VH SAR bands and passed through TerraMind-1.0-small with trained task heads for vessel probability, fishing probability, and vessel length.
3. Calibration and robustness: scene inference uses the validation-calibrated vessel threshold stored in the checkpoint. Optional test-time augmentation averages predictions over flip/transpose variants.
4. AIS fusion: detections are compared against a local AIS CSV cache. Detections without a nearby AIS contact become dark-vessel alerts.

The model path is TerraMind-first. The smaller SAR-CNN is retained only as a baseline.

## Data

Training uses the xView3 maritime SAR metadata and extracted Sentinel-1 scene rasters:

- `train.csv` and `validation.csv` provide labeled vessel candidates.
- `VV_dB.tif` and `VH_dB.tif` provide SAR image bands.
- The trainer filters to locally available scenes and skips unreadable/corrupt rasters before training.

xView3 supports vessel detection, fishing classification, and length estimation. It does not provide tanker/cargo/naval labels, so this submission does not claim those classes.

## The Results :D

The preferred final checkpoint is `artifacts/terramind_trainval/best.pt`. It was trained on the locally available train+validation scene pool with a fresh scene-level holdout, so validation scenes used for scoring were not also used for training.

| Method | Data Split | Vessel F1 @ 0.5 | Best Vessel F1 | Fishing F1 | Length MAE |
|---|---|---:|---:|---:|---:|
| SAR-CNN baseline | tiny | 0.367 | 0.367 | 0.421 | 40.9 m |
| TerraMind-small fine-tuned | tiny | 0.822 | 0.822 | 0.441 | 39.1 m |
| TerraMind-small fine-tuned | larger available subset | 0.637 | 0.691 | 0.468 | 37.4 m |
| TerraMind-small fine-tuned | train+validation, fresh holdout | 0.852 | 0.855 | 0.845 | 26.8 m |
| TerraMind-small resumed fine-tune | train+validation, fresh holdout | 0.848 | 0.851 | 0.830 | 26.3 m |

The resumed fine-tuning run improved length error slightly but did not beat the primary checkpoint on vessel or fishing F1, so the package keeps `terramind_trainval` as the final model.

## Demo

Fast AIS-fusion demo using included mock detections:

```powershell
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
```

Real-scene TerraMind inference, when extracted xView3 scenes are available:

```powershell
python infer.py --scene-id 05bc615a9b0e1159t --scene-root <SCENE_ROOT> --checkpoint artifacts\terramind_trainval\best.pt --detections-output out\terramind_trainval_tta_scene_detections.csv --max-candidates 128 --batch-size 2 --overview-max-dim 1024 --tta --tta-variants 4
```

Replace `<SCENE_ROOT>` with the local path containing extracted xView3 scene folders such as `05bc615a9b0e1159t/VV_dB.tif` and `05bc615a9b0e1159t/VH_dB.tif`. On the original training machine this path was `D:\full`; on another machine it may be different.

## To Run This

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the final model configuration:

```powershell
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root <SCENE_ROOT> --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts\terramind_trainval --overwrite-output --scene-read-probe-size 256
```

Replace `<SCENE_ROOT>` with the actual extracted xView3 scene root on the machine running training. Do not use `D:\full` unless the scenes are actually stored there.

Optional resumed fine-tuning:

```powershell
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root <SCENE_ROOT> --resume-checkpoint artifacts\terramind_trainval\best.pt --epochs 5 --batch-size 2 --lr 3e-5 --backbone-lr 2e-6 --output-dir artifacts\terramind_trainval_long --overwrite-output --scene-read-probe-size 256
```

Replace `<SCENE_ROOT>` with the same extracted xView3 scene root used for the primary training run.

## Submitted Contents

The final `Fantastic-Four` folder contains source code, requirements, sample inputs and outputs, metrics, generated detection CSVs, and the selected checkpoint excludinng the xView3 rasters and downloaded scenes.

## Limitations :(

- Dark-vessel status is not a direct xView3 label. It is inferred by fusing SAR detections with AIS non-matches.
- AIS matching uses a local CSV cache for the demo, not a live AIS feed.
- xView3 does not provide tanker/cargo/naval labels.
- More full-dataset training would likely improve generalization.
- Test-time augmentation improves robustness but increases inference cost.
