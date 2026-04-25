# Ghost Fleet Trigger: Presentation Outline

## Slide 1 - Mission Need

Coast guards and sanctions-enforcement teams need a short list of suspicious vessels, not raw satellite scenes. AIS-only monitoring fails when vessels disable or spoof their transponders. SAR imagery provides an independent sensing layer that works at night and through cloud cover.

Key claim: the system downlinks actionable targets instead of gigabytes of imagery.

## Slide 2 - System Architecture

Input: xView3 Sentinel-1 SAR scenes with VV/VH bands.

Pipeline:

- SAR proposal generator identifies bright, high-contrast, small-object candidates.
- TerraMind-1.0-small scores each crop for vessel probability, fishing probability, and length.
- Calibrated thresholding converts scores into candidate vessel detections.
- Local AIS matching suppresses vessels that already have a nearby AIS contact.
- The downlink payload is a compact dark-vessel alert.

## Slide 3 - Why TerraMind

TerraMind is a geospatial foundation model, so the solution starts from a pretrained Earth-observation representation instead of a random CNN. The submission fine-tunes TerraMind-small on xView3 SAR crops and keeps a SAR-CNN only as a comparison baseline.

Operational fit:

- TerraMind-small is compact enough for the track.
- The checkpoint is under the 200 MB package limit.
- Inference returns a small alert JSON/CSV payload.

## Slide 4 - Results

Preferred checkpoint: `artifacts/terramind_trainval/best.pt`.

| Method | Data Split | Vessel F1 | Fishing F1 | Length MAE |
|---|---|---:|---:|---:|
| SAR-CNN baseline | tiny | 0.367 | 0.421 | 40.9 m |
| TerraMind fine-tuned | tiny | 0.822 | 0.441 | 39.1 m |
| TerraMind fine-tuned | larger available subset | 0.637 | 0.468 | 37.4 m |
| TerraMind fine-tuned | train+validation, fresh holdout | 0.852 | 0.845 | 26.8 m |
| TerraMind resumed fine-tune | train+validation, fresh holdout | 0.848 | 0.830 | 26.3 m |

Interpretation: the resumed run slightly improves length error, but the primary checkpoint remains the best vessel/fishing detector and is the one packaged for submission.

## Slide 5 - Demo Flow

Command for the fast AIS-fusion demo:

```powershell
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
```

Expected story:

- The model detection list contains candidate vessels.
- The AIS cache contains known cooperative vessels.
- A detection without a nearby AIS match becomes the downlink alert.

Real-scene command if local xView3 scenes are present:

```powershell
python infer.py --scene-id 05bc615a9b0e1159t --scene-root D:\full --checkpoint artifacts\terramind_trainval\best.pt --detections-output out\terramind_trainval_tta_scene_detections.csv --max-candidates 128 --batch-size 2 --overview-max-dim 1024 --tta --tta-variants 4
```

## Slide 6 - Honest Limits And Next Steps

Current limits:

- xView3 supports vessel/fishing/length labels, not tanker/cargo/naval labels.
- Dark-vessel status is produced by AIS fusion, not directly by the model.
- The demo uses a local AIS CSV cache, not a live AIS stream.
- The local training set is still a subset of the full xView3 release.

Next steps:

- Complete full-scene xView3 training.
- Add real AIS ingestion and temporal vessel tracking.
- Tune operating threshold for customer preference: recall-first coast guard mode or precision-first analyst mode.
