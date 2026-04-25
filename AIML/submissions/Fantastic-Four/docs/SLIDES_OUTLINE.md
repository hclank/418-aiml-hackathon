# 5-Minute Presentation Outline

## Slide 1 - Customer

Coast guards and sanctions-enforcement teams need a short list of suspicious vessels now. Dark ships turn off AIS, so AIS-only monitoring misses the highest-risk targets.

## Slide 2 - Data And Model

Input is xView3 Sentinel-1 SAR. The final model path uses TerraMind-1.0-small over standardized VV/VH crops, then task heads for vessel probability, fishing probability, and length. The strongest run fine-tunes TerraMind with a lower encoder learning rate and uses the SAR-CNN only as a baseline. A local AIS cache is used after detection to suppress vessels that already have a matching AIS contact.

## Slide 3 - Demo

Run:

```bash
python infer.py --detections sample_input/mock_detections.csv --ais sample_input/mock_ais.csv --output out/alerts.json
```

Show the console output and `out/alerts.json`: one candidate has no nearby AIS match and becomes the downlink alert.

## Slide 4 - Numbers

Use the validation table from `README.md`.

| Method | Data | Vessel F1 | Fishing F1 | Length MAE |
|---|---|---:|---:|---:|
| SAR-CNN baseline | tiny | 0.367 | 0.421 | 40.9 m |
| TerraMind-small fine-tuned | tiny | 0.822 | 0.441 | 39.1 m |
| TerraMind-small fine-tuned | larger subset | 0.637 | 0.468 | 37.4 m |
| TerraMind-small fine-tuned | train + validation scenes | 0.852 | 0.845 | 26.8 m |

Also show:

- Model size: 21.8M parameters, about 41.6 MB at FP16.
- Batch-1 latency: 29.2 ms median on local CPU after cache warmup.
- Bandwidth comparison: xView3 SAR rasters are GB-scale; alert JSON is KB-scale.

## Slide 5 - Limits And Next Week

Honest limits:

- xView3 does not provide tanker/cargo labels, so the demo reports vessel/fishing/length only.
- AIS matching uses a local CSV cache, not a live AIS feed.
- The larger local run still covers only the downloaded subset, not the full xView3 release.

Next week:

- Re-download corrupt/missing xView3 archives and train on the full release.
- Calibrate operating threshold for high recall.
- Add temporal tracking across multiple passes.
