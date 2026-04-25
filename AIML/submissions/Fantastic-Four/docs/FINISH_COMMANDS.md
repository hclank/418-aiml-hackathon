# Reproducibility Commands

Run all commands from `AIML/submissions/Fantastic_Four` unless noted otherwise.

## Validate Code

```powershell
python -B -m py_compile train.py infer.py profile_model.py check_submission.py src\ghost_fleet\model.py src\ghost_fleet\proposals.py
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
python profile_model.py --backbone terramind-small --no-terramind-pretrained --runs 3
```

## Extract xView3 Scenes

Training expects extracted scenes under one scene root, for example `D:\full`.

```powershell
python -B extract_xview3.py --archives-dir D:\full\downloaded_more --output-dir D:\full
python -B extract_xview3.py --archives-dir D:\full\validation --output-dir D:\full
python -B verify_xview3.py --scene-root D:\full --train-csv data\xview3\train.csv --validation-csv data\xview3\validation.csv
```

The extractor skips already extracted scenes and reports corrupt archives without stopping the whole batch.

## Primary Training Run

This is the preferred final configuration. It merges the official train and validation metadata, creates a fresh scene-level holdout, filters to locally available scenes, and skips unreadable rasters.

```powershell
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root D:\full --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts\terramind_trainval --overwrite-output --scene-read-probe-size 256
```

## Optional Resumed Fine-Tuning

This continues from the selected checkpoint. The completed resumed run slightly improved length MAE but did not improve the main vessel/fishing F1 metrics, so it is not the packaged default.

```powershell
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root D:\full --resume-checkpoint artifacts\terramind_trainval\best.pt --epochs 5 --batch-size 2 --lr 3e-5 --backbone-lr 2e-6 --output-dir artifacts\terramind_trainval_long --overwrite-output --scene-read-probe-size 256
```

## Real-Scene Inference

Scene inference uses the checkpoint's calibrated vessel threshold when `--min-vessel-score` is omitted. `--tta` averages predictions over flip/transpose variants.

```powershell
python infer.py --scene-id 05bc615a9b0e1159t --scene-root D:\full --checkpoint artifacts\terramind_trainval\best.pt --detections-output out\terramind_trainval_tta_scene_detections.csv --max-candidates 128 --batch-size 2 --overview-max-dim 1024 --tta --tta-variants 4
```

## Package Submission

```powershell
powershell -ExecutionPolicy Bypass -File prepare_submission.ps1 -Force -Zip
python -B ..\Fantastic-Four\check_submission.py ..\Fantastic-Four
```

The prepared `Fantastic-Four` folder excludes raw xView3 rasters and downloaded archives.
