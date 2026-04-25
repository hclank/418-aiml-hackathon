# Finish Commands

Run these from `AIML/submissions/Fantastic_Four`.

## Local Smoke Checks

Run these in the working folder before packaging. `py_compile` writes
`__pycache__`, so rerun `prepare_submission.ps1` after syntax checks.

```powershell
python -B -m py_compile train.py infer.py profile_model.py check_submission.py src\ghost_fleet\model.py
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
python profile_model.py --backbone terramind-small --no-terramind-pretrained --runs 3
```

## GPU Training Run

Use Colab/Kaggle/local CUDA. Do not run this on CPU unless you only want to test failure paths.

```bash
pip install -r requirements.txt
python train.py --backbone terramind-small --finetune-terramind --available-only --scene-root D:/full --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts/terramind_moredata --overwrite-output --scene-read-probe-size 256
python profile_model.py --checkpoint artifacts/terramind_moredata/best.pt --runs 30
```

This run uses only `vv vh` by default, standardizes them for TerraMind's Sentinel-1 GRD encoder, and fine-tunes TerraMind with a lower encoder learning rate. It also probes the extracted SAR rasters first and skips corrupt scenes. Copy the final metrics from `artifacts/terramind_moredata/metrics.jsonl` into the README table.

## Train With Downloaded Validation Scenes

If the validation archives are under `D:\full\validation`, extract them into the same scene root first:

```powershell
python -B extract_xview3.py --archives-dir D:\full\validation --output-dir D:\full
```

Then train on the union of `train.csv` and `validation.csv`, while creating a new scene-level holdout for metrics:

```powershell
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root D:\full --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts\terramind_trainval --overwrite-output --scene-read-probe-size 256
```

## Baseline Run

```bash
python train.py --backbone sar-cnn --available-only --epochs 5 --batch-size 32 --lr 5e-4 --output-dir artifacts/sar_cnn_baseline
```

## Prepare Submission Folder

```powershell
powershell -ExecutionPolicy Bypass -File prepare_submission.ps1 -Force
python -B ..\Fantastic-Four\check_submission.py ..\Fantastic-Four
```

The prepared `Fantastic-Four` folder excludes extracted xView3 scenes and downloaded archives.
