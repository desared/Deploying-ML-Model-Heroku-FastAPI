# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML model deployment project that trains a RandomForest classifier on Census income data and serves predictions via a FastAPI REST API. The model predicts whether income exceeds $50K/year based on census attributes.

## Common Commands

### Run the API Server Locally
```bash
uvicorn main:app --reload
```

### Train the Model
```bash
python train.py
```

Run specific pipeline steps:
```bash
python train.py main.steps=data_cleaning
python train.py main.steps=train_model
python train.py main.steps=check_score
```

### Run Tests
```bash
pytest
```

Run a single test file:
```bash
pytest test_case/test_api_server.py
```

### Linting
```bash
flake8 --max-line-length 99
```

### DVC Commands
```bash
dvc pull      # Pull data/models from remote storage
dvc push      # Push data/models to remote storage
```

## Architecture

### API Layer (`main.py`, `schema.py`)
- FastAPI application with GET (welcome message) and POST (inference) endpoints
- `ModelInput` Pydantic model validates inference requests with strict Literal types for categorical fields
- Field name mapping handled via `config.yml` (converts Python-friendly names like `marital_status` to dataset names like `marital-status`)

### Training Pipeline (`train.py`, `training/`)
- Uses Hydra for configuration management (`config.yml`)
- Three pipeline stages: `data_cleaning` -> `train_model` -> `check_score`
- `training/modelling/data.py`: Data preprocessing with OneHotEncoder for categoricals, LabelBinarizer for labels
- `training/modelling/model.py`: RandomForestClassifier with 10-fold stratified cross-validation
- `training/val_model.py`: Model validation with per-category slice metrics

### Inference (`training/inferance_model.py`)
- Loads serialized model artifacts from `model/` directory (model.joblib, encoder.joblib, lb.joblib)
- Applies same preprocessing pipeline used during training

### Data Flow
1. Raw data: `data/census.csv`
2. Cleaned data: `data/clean_census.csv` (removes nulls, duplicates, drops education-num/capital columns)
3. Model artifacts: `model/model.joblib`, `model/encoder.joblib`, `model/lb.joblib`

## Configuration

All configuration is in `config.yml`:
- `data.cat_features`: List of categorical feature column names
- `infer.update_keys`: Mapping from API field names to dataset column names
- `infer.columns`: Expected column order for inference

## Deployment

- Heroku deployment via `Procfile` using uvicorn
- DVC automatically pulls data on Heroku startup (see conditional in `main.py`)
- GitHub Actions runs flake8 and pytest on push to main
- AWS S3 used as DVC remote storage
