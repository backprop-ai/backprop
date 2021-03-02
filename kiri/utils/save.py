from kiri.models import BaseModel
import dill
import pkg_resources
import os
import json

def save(model, path=None):
    name = model.name
    tasks = model.tasks
    description = model.description

    if path is None and name is None:
        raise ValueError("please provide a path or give the model a name")

    if path is None:
        path = os.path.expanduser(f"~/.cache/kiri/{name}")

    os.makedirs(path, exist_ok=True)

    config = {
        "description": description,
        "tasks": tasks
    }

    packages = ["kiri", "transformers", "sentence_transformers"]
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