from backprop.models import HFNLIModel
from backprop import load
import torch

class AutoModel:
    @staticmethod
    def from_pretrained(model_name, device=None):
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = None

        models = AutoModel.list_models()
        model_config = models.get(model_name)

        if model_config == None:
            try:
                model = load(model_name)
                model.to(device)
            except ValueError:
                raise ValueError(f"Model '{model_name}' not found")
    
        if model == None:
            model_class = model_config["class"]
            init_kwargs = model_config["init_kwargs"]

            model = model_config["class"](**init_kwargs, device=device)

        return model

    @staticmethod
    def list_models(task=None):
        models_classes = [HFNLIModel]

        models_list = {}
        for model_class in models_classes:
            models = model_class.list_models()
            
            for model_name, model_config in models.items():
                if not task or task in model_config.get("tasks"):
                    model_config["class"] = model_class
                    models_list[model_name] = model_config

        return models_list
