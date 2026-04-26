ThatSlurp
ThatSlurp
thatslurp
Online

teknightmare [dih],  — Yesterday at 10:41 PM
# Judging Rubric

How submissions are scored. Read this *before* you start building — it'll tell you where to spend your last 12 hours.

## The rubric

JUDGING.md
6 KB
teknightmare [dih],  — Yesterday at 11:09 PM
https://claude.ai/share/553aee83-9047-4c40-a762-eea8f110850f
is the project uploaded yet?
@ThatSlurp do add a readme.md of the model itself too
teknightmare [dih],  — 12:19 AM
link the datasets
teknightmare
 changed the group name: Fantastic 4. Edit Group — 12:25 AM
ThatSlurpThatSlurp — 12:26 AM
Here is the beginner-level mental model you need.

**One-Line Pitch**
We built a satellite-based “dark vessel” alert system. TerraMind looks at SAR satellite imagery, detects likely vessels, estimates whether they are fishing vessels and their length, then compares detections against AIS ship positions. If a vessel appears in imagery but has no nearby AIS signal, we flag it as a suspicious dark vessel.

**The Problem**

message.txt
6 KB
teknightmare [dih],  — 01:02 AM
in the 2 minute suggestion they gave
it says one slide one customer one sentence tbh
😭
we have too many customers dont we?
scoop [PKMN],  — 01:07 AM
parth look up nga
 [PKMN], 
teknightmare [dih],  — 01:10 AM
Image
teknightmare [dih],  — 01:22 AM
update the ppt according to the updated readme
teknightmare [dih],  — 05:50 AM
Attachment file type: document
ghost_fleet_trigger.pptx
351.48 KB
scoop [PKMN],  — 06:11 AM
Ghost Fleet Trigger
Dark-vessel alerts from orbital SAR - Downlink the answer not the data

TL;DR
Ghost Fleet Trigger is a Terramind-powered SAR pipeline that converts xView3 imagery into actionable dark-vessel alerts.

It detects VV/VH radar data, estimates fishing likelyhood and vessel length, cross checks with a local AIS cache, and only spits out high-risk targets

Why It Matters
Coast-guard maritime-surveillance teams need to identify dark-vessels with minutes of a Sentinel-1 pass - without having to downlink or manually review gigabytes of raw SAR imagery
scoop [PKMN],  — 06:23 AM
Core Idea
Take SAR Input from xView3 and find vessel locations with a TerraMind based detector.
for each detection, estimate the confidence, fishing likelyhood and vessel length.
Compare each detection with the local AIS cache for nearby known ships.
If there is no AIS match then we can label it as a possible dark-vessel and generate an alert with the coordinates, risk label and confidence
teknightmare [dih],  — 06:44 AM
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

Prepare & clean final folder:

```powershell
powershell -ExecutionPolicy Bypass -File prepare_submission.ps1 -Force
```

Before uploading. Run:

```bash
python -B check_submission.py ../Fantastic-Four
```


README.md
6 KB
ThatSlurpThatSlurp — 06:51 AM
oh sht wait no
﻿
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