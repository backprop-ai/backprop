from . import save
from kiri.models import BaseModel
from zipfile import ZipFile
import os
import dill
import requests

def upload(model: BaseModel = None, path: str = None, api_key: str = None, save_path: str = None):
    """
    Deploys a model from object or path to Kiri. 
    Either the model or path to saved model must be provided.

    Args:
        model: Model object
        path: Path to saved model
        api_key: Kiri API key
        save_path: Optional path to save model if providing a model object
    """
    
    if api_key is None:
        raise ValueError("Please provide your api_key")

    if model is None and path is None:
        raise ValueError("You must either specify the model or path to saved model")

    if model is not None and path is not None:
        raise ValueError("You may only specify either the model or the path to saved model")

    if model:
        print("Saving model...")
        if hasattr(model, "to"):
            model.to("cpu")
        path = save(model, path=save_path)

    print("Testing that the model can be loaded...")
    # Loading model to get the model name
    with open(os.path.join(path, "model.bin"), "rb") as f:
        model = dill.load(f)
        model_name = model.name

    # Save working directory
    cwd = os.getcwd()

    # Move to model directory
    os.chdir(path)

    print("Creating zip...")
    zip_obj = ZipFile(f"{model_name}.zip", "w")
    
    zip_obj.write("config.json")
    zip_obj.write("requirements.txt")
    zip_obj.write("model.bin")

    zip_obj.close()


    print("Getting upload url...")
    res = requests.post("https://api.kiri.ai/upload-url",
                                json={"model_name": model_name},
                                headers={"x-api-key": api_key})
    if res.status_code != 200:
        out = res.json().get("message")

        if out is None:
            out = res.json().get("error")

        raise Exception(f"Failed to get upload url: {out}")
    
    upload_url = res.json()

    print("Uploading to Kiri, this may take a few minutes...")
    with open(f"{model_name}.zip", "rb") as f:
        res = requests.put(upload_url, f)

    if res.status_code != 200:
        raise Exception(f"Failed to upload. Please try again.")

    print("Successfully uploaded the model to Kiri. See the build process at https://dashboard.kiri.ai")

    # Move back to working directory
    os.chdir(cwd)