import pytest
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation

def test_feature_engineering_logic():
    # Setup dummy data
    data = {
        'tenure': [1, 2, 3],
        'TotalCharges': [100.0, 200.0, 300.0],
        'SeniorCitizen': [0, 1, 0],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'InternetService': ['Fiber optic', 'No', 'DSL'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        # Missing other service columns to verify robustness
    }
    df = pd.DataFrame(data)
    
    dt = DataTransformation()
    transformed_df = dt.get_feature_engineered_df(df.copy())
    
    # Check if new columns are created
    assert 'AvgMonthlyCharge' in transformed_df.columns
    assert 'ServiceCount' in transformed_df.columns
    
    # Check calculation logic (tenure + 1)
    # Row 0: 100 / (1 + 1) = 50.0
    assert transformed_df.loc[0, 'AvgMonthlyCharge'] == 50.0
    
    # Check SeniorCitizen conversion
    assert isinstance(transformed_df.loc[0, 'SeniorCitizen'], str)
    
    # Check ServiceCount logic
    # Row 0: PhoneService('Yes'=1) + InternetService('Fiber'!=No=1) + OnlineSecurity('No'!=Yes=0) = 2.0
    assert transformed_df.loc[0, 'ServiceCount'] == 2.0

def test_get_data_transformer_object():
    dt = DataTransformation()
    preprocessor = dt.get_data_transformer_object()
    
    # Check if it's a ColumnTransformer
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Check if it has the expected number of transformers
    assert len(preprocessor.transformers) == 2 # num and cat
