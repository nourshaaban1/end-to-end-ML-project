# Customer Churn Prediction - End-to-End ML Project

A comprehensive machine learning project for predicting customer churn using telecommunications data. This project implements a complete end-to-end ML pipeline with data ingestion, transformation, model training, and a Flask-based web application for predictions.

## Project Overview

This project demonstrates a production-ready ML workflow including:

- **Data Ingestion**: Loads and splits data into train/validation/test sets
- **Feature Engineering**: Creates new features (AvgMonthlyCharge, ServiceCount)
- **Preprocessing**: Handles categorical variables, missing values, and class imbalance using SMOTENC
- **Model Training**: Evaluates multiple models (RandomForest, HistGradientBoosting) with cross-validation and GridSearchCV hyperparameter tuning
- **Web Application**: Flask-based UI for real-time predictions

## Architecture

```
src/
├── components/
│   ├── data_ingestion.py       # Data loading and splitting
│   ├── data_transformation.py  # Feature engineering & preprocessing
│   └── model_trainer.py        # Model training & tuning
├── pipeline/
│   ├── train_pipeline.py       # Training orchestration
│   └── predict_pipeline.py     # Prediction pipeline
├── utils.py                    # Helper functions
├── logger.py                   # Logging configuration
└── exception.py                # Custom exception handling
```

## Data Flow

```
data/train.csv
    ↓
DataIngestion → train.csv, validation.csv, test.csv in artifacts/
    ↓
DataTransformation (feature engineering + SMOTENC + preprocessing)
    ↓
train_arr, validation_arr, test_arr
    ↓
ModelTrainer (evaluate → select best → GridSearchCV → save model.pkl)
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd end-to-end-ML-project

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

## Usage

### Training the Model

```bash
python -m src.pipeline.train_pipeline
```

This will:
1. Load and split the data
2. Apply feature engineering and preprocessing
3. Evaluate multiple models
4. Perform hyperparameter tuning with GridSearchCV
5. Save the best model and preprocessing artifacts to `artifacts/`

### Running Predictions

```bash
python -m src.pipeline.predict_pipeline
```

### Running the Web Application

```bash
python application.py or python application
```

Then open your browser to `http://localhost:5000` to use the prediction interface.

The application provides:
- **Web UI**: Interactive form for manual customer data entry
- **API Endpoint**: `POST /predict-api` for JSON-based predictions

## Configuration

Key configuration paths (defined in components):

| Component | Output |
|-----------|--------|
| Data Ingestion | `artifacts/train.csv`, `artifacts/validation.csv`, `artifacts/test.csv` |
| Data Transformation | `artifacts/preprocessor.pkl`, `artifacts/label_encoder.pkl` |
| Model Training | `artifacts/model.pkl` |

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
seaborn
jupyterlab
lightgbm
dill
imblearn
flask
```

## Model Information

The project evaluates multiple algorithms and selects the best performing model based on cross-validation scores. Current metrics (after GridSearchCV):

- **RandomForest**: High accuracy with good generalization
- **HistGradientBoosting**: Fast training with competitive accuracy

## Project Structure

- `src/`: Source code
- `data/`: Dataset (train.csv)
- `artifacts/`: Saved models and preprocessors
- `templates/`: Flask HTML templates
- `logs/`: Training and prediction logs

## Docker Deployment

Build and run the Docker image:

```bash
# Build the image
docker build -t churn-prediction .

# Run the container
docker run -p 5000:5000 churn-prediction
```

The application will be available at `http://localhost:5000`.

**Note**: The model artifacts must be present in the `artifacts/` directory before running the application. Run the training pipeline first to generate these files.

## Author

Nour Shaaban - nourshaaban59@gmail.com

