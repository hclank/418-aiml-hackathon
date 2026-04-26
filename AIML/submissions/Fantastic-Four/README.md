## Ghost Fleet Trigger - Team Fantastic Four

Ghost Fleet Trigger converts xView3 Sentinel 1 SAR ship detections to minimal dark-vessel alerts for the maritime security. 

"Fantastic-Four" is the final submission folder.


## 1. What problem are we solving?

The project's target customer is a coast guard or a commodity-risk team that requires to find vessels that are worthy of investigation instead of browsing through raw SAR imagery. Ships that turn off or spoof AIS often tend to be involved in illegal fishing and sanctioned oil transfers. The target output of this project is a concise message like: vessel detected at "coordinates", no AIS match nearby, vessel confidence, fishing probablity and estimated length. This is valuable because it lets analysts request another image, flag a transfer or task a patrol using only a couple hundred bytes instead of processing gigabytes of imagery.

## 2. What did we build?

Using TerraMind-small as the main model path, we built an `xView3 SAR pipeline`. 

This model takes VV / VH crops from xView3 -> standardizes them to Sentinel 1 distribution -> runs them through TerraMind 1.0 small ->
trains task heads for the non vessel/vessel, fishing/non fishing, and length.

The model's path is TerraMind-first; the small SAR-CNN is retained only to be the baseline.

xView3 lets us perform vessel detection, length estimation and fishing classification. However, it cannot provide tanker/naval/cargo labels, so this project does not claim those classes.

Training commands that were used for the larger TerraMind run:


```bash
python train.py --backbone terramind-small --finetune-terramind --available-only --scene-root <scene_location> --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts/terramind_moredata --overwrite-output --scene-read-probe-size 256
```

In the case validation scenes are available locally, trainer can fold train.csv & validation.csv into a single labeled pool and then make a new scene-level holdout:

```bash
python train.py --backbone terramind-small --finetune-terramind --train-on-validation --available-only --scene-root <scene_location> --epochs 10 --batch-size 2 --lr 7e-5 --backbone-lr 5e-6 --output-dir artifacts/terramind_trainval --overwrite-output --scene-read-probe-size 256
```

Fast demo command:

```bash
python infer.py --detections sample_input/mock_detections.csv --ais sample_input/mock_ais.csv --output out/alerts.json
```

replace <scene_location> with the actual location of all the scenes

## 3. How did we measure it?

The first run uses xView3 tiny dataset. The larger run uses local xView3 subset. The trainer probes SAR rasters before the training and skips partially corrupt scenes.

| Method | Data Split | Vessel F1 @ 0.5 | Best Vessel F1 | Fishing F1 | Length MAE |
|---|---|---:|---:|---:|---:|
| SAR-CNN baseline | tiny | 0.367 | 0.367 | 0.421 | 40.9 m |
| TerraMind-small fine-tuned | tiny | 0.822 | 0.822 | 0.441 | 39.1 m |
| TerraMind-small fine-tuned | larger available subset | 0.637 | 0.691 | 0.468 | 37.4 m |
| TerraMind-small fine-tuned | train+validation, fresh holdout | 0.852 | 0.855 | 0.845 | 26.8 m |
| TerraMind-small resumed fine-tune | train+validation, fresh holdout | 0.848 | 0.851 | 0.830 | 26.3 m |

 the baseline CNN is more conservative and gets higher precision, but TerraMind is much better for the actual coast-guard use case because it catches far more vessels. Recall jumps from 0.517 to 0.922, which means it misses far fewer ships.

TerraMind-small run is the main model path. The training and validation run is the preferrable checkpoint. We can also measure the operational output: detections are made into AIS-unmatched alerts and the demo command gives one dark-vessel-style alert from 3 candidates and 2 AIS contacts.

## 4. What is the oribital compute story?

It would be very inefficient for the satellite to downlink every SAR scene for a human to search. So instead, it should perform ship triage on the satellite, check the detections against the daily AIS cache and only downlink suspicious targets. xView3 scenes in the tiny set have VV / VH rasters of over 1 GB per band. One alert JSON is less than 1 KB. So, the main `downlink the answer, not the data` story lies here :). TerraMind-small is the right scale for the track: this run has 21.8 M parameters, about 41.6 MB at FP16 and measured 29.2 ms median batch-1 latency on local CPU after the model was cached.

## 5. What does not work YET?

Actually, the data we currently have is from the xView3 SAR dataset, which means that the model is only capable of claiming the detection of vessels, the probability of fishing, and the estimation of the length. It cannot directly claim the tanker/cargo/patrol class because xView3 did not provide those labels. Instead of a live AIS stream, demonstration of AIS matching is done with a local CSV cache,. Training with the original script was interrupted several times because some downloaded archives were incomplete. Now the script just skips the unreadable raster when it comes across it. Next step is getting the data for the remaining scenes, and setting the alert threshold for coast-guard recall.

## Reproducibility and submission

Install & run fast demo:

```bash
pip install -r requirements.txt
python infer.py --detections sample_input/mock_detections.csv --ais sample_input/mock_ais.csv --output out/alerts.json
```

Real-scene TerraMind inference, when extracted xView3 scenes are available:

```bash
python infer.py --scene-id 05bc615a9b0e1159t --scene-root <SCENE_ROOT> --checkpoint artifacts\terramind_trainval\best.pt --detections-output out\terramind_trainval_tta_scene_detections.csv --max-candidates 128 --batch-size 2 --overview-max-dim 1024 --tta --tta-variants 4
```

## Links to Datasets

xView3: https://iuu.xview.us/download-links
