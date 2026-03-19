# ATLAS

# How to Use

---

## Step 1: Set the event log in config.yaml and run the stream analysis.
```
python -m src.main_atlas_new --config config.yaml --log helpdesk
```

## Step 2: Model Evaluation
```
python -m src.cm_metrics --log helpdesk
```
