import pytest
import os
import shutil
from pathlib import Path
from src.utils import save_object, load_object

def test_save_and_load_object(tmp_path):
    # Setup
    test_obj = {"key": "value", "list": [1, 2, 3]}
    file_path = tmp_path / "test_artifact" / "model.pkl"
    
    # Test saving
    save_object(file_path=file_path, obj=test_obj)
    
    # Check if file exists
    assert os.path.exists(file_path)
    
    # Test loading
    loaded_obj = load_object(file_path=file_path)
    
    # Check if content is same
    assert loaded_obj == test_obj
    assert loaded_obj["key"] == "value"
    assert len(loaded_obj["list"]) == 3

def test_save_object_exception():
    # Test handling of invalid path (this should trigger CustomException)
    from src.exception import CustomException
    with pytest.raises(CustomException):
        # Using a path that's likely invalid or permission denied
        save_object(file_path="/invalid_root_path/test.pkl", obj={})
