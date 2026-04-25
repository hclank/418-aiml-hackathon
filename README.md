# 418 AIML Hackathon

Final Ghost Fleet/TerraMind submission package:

```text
AIML/submissions/Fantastic-Four
```

That folder contains the runnable code, requirements, docs, sample inputs, validation scripts, and the final `terramind_trainval` checkpoint. The raw xView3 scene rasters and downloaded archives are intentionally not committed.

Main entry points:

```powershell
cd AIML\submissions\Fantastic-Four
pip install -r requirements.txt
python infer.py --detections sample_input\mock_detections.csv --ais sample_input\mock_ais.csv --output out\alerts.json
python -B check_submission.py .
```
