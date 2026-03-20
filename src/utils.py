import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import dill
import random
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def univariate_analysis(df):
    results = []

    for col in df.columns:
        col_data = df[col]
        info = {
            'feature': col,
            'count': col_data.count(),
            'missing': col_data.isna().sum(),
            'unique': col_data.nunique(),
            'dtype': col_data.dtype,
            'is_numeric': pd.api.types.is_numeric_dtype(col_data)
        }

        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                'min':    col_data.min(),
                'max':    col_data.max(),
                'mean':   col_data.mean(),
                'median': col_data.median(),
                'mode':   col_data.mode()[0],
                'std':    col_data.std(),
                'q25':    col_data.quantile(0.25),
                'q50':    col_data.quantile(0.50),
                'q75':    col_data.quantile(0.75),
                'skew':   col_data.skew(),
                'kurt':   col_data.kurt()
            })
        else:
            info.update({
                'min': None, 'max': None, 'mean': None, 'median': None,
                'mode': col_data.mode()[0], 'std': None,
                'q25': None, 'q50': None, 'q75': None,
                'skew': None, 'kurt': None
            })

        results.append(info)

    summary = pd.DataFrame(results).set_index('feature')

    # Display formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("=" * 60)
    print("UNIVARIATE ANALYSIS SUMMARY")
    print("=" * 60)
    print(summary.T.to_string())
    print("=" * 60)

    return results

def analyze_bivariate(df, target_column):
    """
    Performs detailed bivariate analysis between all features and a target column.
    Handles Num-Num, Cat-Cat, and Num-Cat relationships.
    """
    features = [col for col in df.columns if col != target_column]
    target_type = 'num' if pd.api.types.is_numeric_dtype(df[target_column]) else 'cat'

    for col in features:
        col_type = 'num' if pd.api.types.is_numeric_dtype(df[col]) else 'cat'
        print(f"\n{'='*20} Analyzing: {col} vs {target_column} {'='*20}")

        plt.figure(figsize=(10, 5))

        # --- CASE 1: NUMERICAL VS NUMERICAL ---
        if col_type == 'num' and target_type == 'num':
            sns.scatterplot(data=df, x=col, y=target_column, alpha=0.5)
            sns.regplot(data=df, x=col, y=target_column, scatter=False, color='red')

            pearson_r, p_val = stats.pearsonr(df[col].dropna(), df[target_column].dropna())
            print(f"Metric: Pearson Correlation = {pearson_r:.4f} (p-value: {p_val:.4e})")
            plt.title(f"Scatter Plot: {col} vs {target_column} (r={pearson_r:.2f})")

        # --- CASE 2: CATEGORICAL VS CATEGORICAL ---
        elif col_type == 'cat' and target_type == 'cat':
            contingency_table = pd.crosstab(df[col], df[target_column])
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')

            chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
            # Cramer's V Calculation
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

            print(f"Metric: Chi-Square = {chi2:.2f} (p-value: {p:.4e})")
            print(f"Metric: Cramer's V = {cramers_v:.4f}")
            plt.title(f"Heatmap: {col} vs {target_column}")

        # --- CASE 3: NUMERICAL VS CATEGORICAL (and vice versa) ---
        else:
            # Determine which is which for the plot
            num_col = col if col_type == 'num' else target_column
            cat_col = target_column if col_type == 'num' else col

            sns.boxplot(data=df, x=cat_col, y=num_col)

            # Statistical Test (ANOVA)
            groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"Metric: ANOVA F-statistic = {f_stat:.2f} (p-value: {p_val:.4e})")
            plt.title(f"Box Plot: {num_col} by {cat_col}")

        plt.tight_layout()
        plt.show()

def save_object(file_path: Path, obj):
    """
    Saves an object to a file using pickle.
    """
    try:
        dir_path: str = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_val, y_val, models, cv_splits=5):
    """
    Trains and evaluates multiple models using Stratified K-Fold Cross Validation.
    Returns a DataFrame with model names as keys and their cross-validation scores as values.
    """
    
    cv_results_list = []
    trained_models = {}
    
    try:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logging.info(f"Training and Evaluating {model_name}...")
            
            # Cross-Validation on X_train
            cv_scores = cross_validate(
                model, 
                X_train, 
                y_train,
                cv=skf,
                scoring=['accuracy', 'f1', 'roc_auc'],
                n_jobs=-1
            )
            
            # Train the final model on the full training set
            model.fit(X_train, y_train)
            trained_models[model_name] = model

            # Validation Evaluation
            y_pred = model.predict(X_val)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_pred_proba = (
                    model.decision_function(X_val)
                    if hasattr(model, "decision_function") else y_pred
                )

            val_acc = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)
            val_roc_auc = roc_auc_score(y_val, y_pred_proba)
            
            # Store in list
            cv_results_list.append({
                'Model': model_name,
                'CV Accuracy': cv_scores['test_accuracy'].mean(),
                'CV F1-Score': cv_scores['test_f1'].mean(),
                'CV ROC-AUC': cv_scores['test_roc_auc'].mean(),
                'Val Accuracy': val_acc,
                'Val F1-Score': val_f1,
                'Val ROC-AUC': val_roc_auc
            })
            
            logging.info(f"{model_name} - Training ROC-AUC: {cv_scores['test_roc_auc'].mean():.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
            
        results_df = pd.DataFrame(cv_results_list)
        results_df = results_df.sort_values(by='Val ROC-AUC', ascending=False).reset_index(drop=True)
        return results_df, trained_models

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: Path):
    """
    Loads an object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
