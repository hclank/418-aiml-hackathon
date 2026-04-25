# 418 AIML Hackathon

Final AIML track submission package:

```text
AIML/submissions/Fantastic-Four
```

The package contains the Ghost Fleet Trigger solution: TerraMind-based SAR vessel scoring, AIS non-match alerting, requirements, sample inputs, metrics, generated detections, validation scripts, and the selected `terramind_trainval` checkpoint. Raw xView3 scene rasters and downloaded archives are intentionally not committed.

Main entry points:

```powershell
cd AIML\submissions\Fantastic-Four
pip install -r requirements.txt
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
python -B check_submission.py .
```
