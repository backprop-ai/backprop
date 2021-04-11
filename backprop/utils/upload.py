from typing import Dict, List
from . import save
from zipfile import ZipFile
import os
import dill
import requests

def upload(model, name: str = None, description: str = None, tasks: List[str] = None,
            details: Dict = None, path=None, api_key: str = None):
    """
    Saves and deploys a model to Backprop.

    Args:
        model: Model object
        api_key: Backprop API key
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Valid json dictionary of additional details about the model
        path: Optional path to save model

    Example::

        import backprop

        tg = backprop.TextGeneration("t5_small")

        # Any text works as training data
        inp = ["I really liked the service I received!", "Meh, it was not impressive."]
        out = ["positive", "negative"]

        # Finetune with a single line of code
        tg.finetune({"input_text": inp, "output_text": out})

        # Use your trained model
        prediction = tg("I enjoyed it!")

        print(prediction)
        # Prints
        "positive"

        # Upload to Backprop for production ready inference

        model = tg.model
        # Describe your model
        name = "t5-sentiment"
        description = "Predicts positive and negative sentiment"

        backprop.upload(model, name=name, description=description, api_key="abc")
    """
    
    if api_key is None:
        raise ValueError("Please provide your api_key")

    print("Saving model...")
    if hasattr(model, "to"):
        model.to("cpu")
    path = save(model, name=name, description=description, tasks=tasks, details=details, path=path)

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
    res = requests.post("https://api.backprop.co/upload-url",
                                json={"model_name": model_name},
                                headers={"x-api-key": api_key})
    if res.status_code != 200:
        out = res.json().get("message")

        if out is None:
            out = res.json().get("error")

        raise Exception(f"Failed to get upload url: {out}")
    
    upload_url = res.json()

    print("Uploading to Backprop, this may take a few minutes...")
    with open(f"{model_name}.zip", "rb") as f:
        res = requests.put(upload_url, f)

    if res.status_code != 200:
        raise Exception(f"Failed to upload. Please try again.")

    print("Successfully uploaded the model to Backprop. See the build process at https://dashboard.backprop.co")

    # Move back to working directory
    os.chdir(cwd)