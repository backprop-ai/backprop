from typing import Dict
from backprop.models import HFNLIModel, STModel, HFSeq2SeqTGModel, T5QASummaryEmotion, \
    HFCausalLMTGModel, HFSeqTCModel, EfficientNet, CLIP
from backprop import load
import torch

class AutoModel:
    @staticmethod
    def from_pretrained(model_name: str, aliases: Dict = None, device: str = None):
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = None

        models = AutoModel.list_models(return_dict=True)
        model_config = models.get(model_name)

        # Try to find by alias
        if model_config == None and aliases:
            name_from_alias = aliases.get(model_name)

            if name_from_alias:
                model_config = models.get(name_from_alias)
                if model_config:
                    model_name = name_from_alias

        # Try to load from local saved model
        if model_config == None:
            try:
                model = load(model_name)
                model.to(device)
            except ValueError:
                raise ValueError(f"Model '{model_name}' not found")
    
        if model == None:
            model_class = model_config["class"]
            init_kwargs = model_config["init_kwargs"]

            model = model_config["class"](**init_kwargs,
                    description=model_config["description"],
                    tasks=model_config["tasks"],
                    name=model_name,
                    details=model_config.get("details"),
                    device=device)

        return model

    @staticmethod
    def list_models(task=None, return_dict=False, display=False, limit=None):
        models_classes = [HFNLIModel, STModel, HFSeq2SeqTGModel, T5QASummaryEmotion,
                          HFCausalLMTGModel, HFSeqTCModel, EfficientNet, CLIP]

        if display:
            return_dict = False

        models_dict = {}
        models_list = []
        output = None

        num_models = 0

        for model_class in models_classes:
            models = model_class.list_models()
            
            for model_name, model_config in models.items():
                if not task or task in model_config.get("tasks"):
                    
                    if limit and num_models >= limit:
                        break

                    if return_dict:
                        model_config["class"] = model_class
                        models_dict[model_name] = model_config
                        output = models_dict
                    else:
                        model_config["name"] = model_name
                        models_list.append(model_config)
                        output = models_list
                    
                    num_models += 1

        if display:
            for model in models_list:

                print(f"{'Name' :25s}{model['name']}")
                print(f"{'Description' :25s}{model['description']}")
                print(f"{'Supported tasks' :25s}{model['tasks']}")
                print("----------")
        else:
            return output
