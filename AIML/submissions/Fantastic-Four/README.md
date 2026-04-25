# Fantastic-Four - Ghost Fleet Trigger

Ghost Fleet Trigger turns xView3 Sentinel-1 SAR ship detections into compact dark-vessel alerts for maritime security. The final submission folder is `Fantastic-Four`; `Fantastic_Four` is the local working folder that may contain downloaded xView3 data.

## 1. What Problem Are We Solving?

The customer is a coast guard, sanctions-enforcement desk, or commodity-risk team that needs to find vessels worth investigating, not browse raw SAR imagery. Illegal fishing and sanctioned oil transfers often involve ships that turn off or spoof AIS. Our target output is a small message like: vessel detected at latitude/longitude, no AIS match nearby, vessel confidence, fishing probability, and estimated length. That is useful because it lets an analyst task a patrol, request another image, or flag a transfer using a few hundred bytes instead of downlinking gigabytes of imagery.

## 2. What Did We Build?

We built an xView3 SAR pipeline with TerraMind-small as the main model path. The model takes VV/VH SAR crops from xView3, standardizes them to the Sentinel-1 distribution expected by TerraMind, runs them through TerraMind-1.0-small, and trains task heads for vessel/non-vessel, fishing/non-fishing, and vessel length. A separate alert layer cross-matches detections against a local AIS CSV cache and emits only unmatched vessel alerts. The non-TerraMind SAR-CNN path remains in the repo only as a baseline for comparison.

Training command used for the larger TerraMind run:

```bash
python train.py --backbone terramind-small --finetune-terramind --available-only --scene-root D:/full --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts/terramind_moredata --overwrite-output --scene-read-probe-size 256
```

If downloaded validation scenes are also available locally, the trainer can fold `train.csv` and `validation.csv` into one labeled pool and create a fresh scene-level holdout:

```bash
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root D:/full --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts/terramind_trainval --overwrite-output --scene-read-probe-size 256
```

Fast demo command:

```bash
python infer.py --detections sample_input/mock_detections.csv --ais sample_input/mock_ais.csv --output out/alerts.json
```

## 3. How Did We Measure It?

The first measured run uses the xView3 tiny set: five training scenes and two validation scenes. The larger run uses the locally available xView3 subset; the trainer probes SAR rasters before training and skips partially extracted/corrupt scenes.

| Method | Data | Vessel F1 @ 0.5 | Best Vessel F1 | Fishing F1 | Length MAE |
|---|---|---:|---:|---:|---:|
| SAR-CNN baseline | tiny | 0.367 | 0.367 | 0.421 | 40.9 m |
| TerraMind-small fine-tuned encoder + heads | tiny | 0.822 | 0.822 | 0.441 | 39.1 m |
| TerraMind-small fine-tuned encoder + heads | larger available subset | 0.637 | 0.691 | 0.468 | 37.4 m |
| TerraMind-small fine-tuned encoder + heads | train + validation scenes with fresh holdout | 0.852 | 0.855 | 0.845 | 26.8 m |

The TerraMind-small run is the primary model path. The train+validation run is the preferred final checkpoint because it uses the most local labeled scenes while still holding out entire scenes for metrics. We also measure the operational output: detections are converted into AIS-unmatched alerts, and the demo command produces one dark-vessel-style alert from three candidate detections and two AIS contacts.

## 4. What Is The Orbital-Compute Story?

The satellite should not downlink every SAR scene for a human to search. It should run ship triage onboard, compare detections against the daily AIS cache, and downlink only suspicious targets. xView3 scenes in the tiny set contain VV/VH rasters over 1 GB per band; one alert JSON is under 1 KB. That is the core "downlink the answer, not the data" story. TerraMind-small is the right scale for the track: this run has 21.8M parameters, about 41.6 MB at FP16, and measured 29.2 ms median batch-1 latency on local CPU after the model was cached.

## 5. What Does Not Work Yet?

The current dataset is xView3 SAR, so the model can honestly claim vessel detection, fishing probability, and length estimation; it cannot directly claim tanker/cargo/patrol class because xView3 does not provide those labels. AIS matching is demonstrated with a local CSV cache, not a live AIS stream. Some downloaded archives were incomplete, so the training script now skips unreadable rasters instead of failing mid-epoch. The next step is downloading the remaining scenes and calibrating the alert threshold for coast-guard recall.

## Reproducibility And Submission

Install and run the fast demo:

```bash
pip install -r requirements.txt
python infer.py --detections sample_input/mock_detections.csv --ais sample_input/mock_ais.csv --output out/alerts.json
```

Prepare the clean final folder from the working folder:

```powershell
powershell -ExecutionPolicy Bypass -File prepare_submission.ps1 -Force
```

Before uploading, run:

```bash
python -B check_submission.py ../Fantastic-Four
```

Do not submit `data/xview3/full/` or `data/xview3/full/downloaded/`; those are local dataset folders only. xView3 source: https://iuu.xview.us/
