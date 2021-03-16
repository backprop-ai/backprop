import dill
import json
import os

def load(path):
    """
    Loads a saved model and returns it.

    Args:
        path: Name of the model or full path to model.
    """
    # Try to look in cache folder
    cache_path = os.path.expanduser(f"~/.cache/kiri/{path}")
    cache_model_path = os.path.join(cache_path, "model.bin")
    if os.path.exists(cache_model_path):
        path = cache_model_path
    else:
        model_path = os.path.join(path, "model.bin")
        
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
            
        if not os.path.exists(model_path):
            raise ValueError("model not found!")

        path = model_path

    with open(os.path.join(path), "rb") as f:
        model = dill.load(f)

    return model