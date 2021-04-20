from typing import List, Dict
import dill
import pkg_resources
import os
import json

dill.settings["recurse"] = True

def save(model, name: str = None, description: str = None, tasks: List[str] = None, details: Dict = None, path=None):
    """
    Saves the provided model to the backprop cache folder using:
        1. provided name
        2. model.name
        3. provided path

    The resulting folder has three files:

        * model.bin (dill pickled model instance)
        * config.json (description and task keys)
        * requirements.txt (exact python runtime requirements)

    Args:
        model: Model object
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Valid json dictionary of additional details about the model
        path: Optional path to save model

    Example::

        import backprop

        backprop.save(model_object, "my_model")
        model = backprop.load("my_model")
    """

    if name:
        model.name = name

    if description:
        model.description = description

    if tasks:
        model.tasks = tasks

    if details:
        model.details = details

    name = model.name
    tasks = model.tasks
    description = model.description
    details = model.details

    if hasattr(model.model, "eval"):
        model.model.eval()

    if path is None and name is None:
        raise ValueError("please provide a path or give the model a name")

    if path is None:
        path = os.path.expanduser(f"~/.cache/backprop/{name}")

    os.makedirs(path, exist_ok=True)

    config = {
        "description": description,
        "tasks": tasks,
        "details": details
    }

    packages = ["backprop", "transformers", "sentence_transformers", "efficientnet_pytorch"]
    requirements = [f"{package}=={pkg_resources.get_distribution(package).version}"
                    for package in packages]

    requirements_str = "\n".join(requirements)

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(path, "requirements.txt"), "w") as f:
        f.write(requirements_str)

    with open(os.path.join(path, "model.bin"), "wb") as f:
        dill.dump(model, f)

    return path