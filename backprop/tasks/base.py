from typing import Dict, List
import logging
from backprop import load

logger = logging.getLogger("info")

class Task:
    def __init__(self, model, local=False, api_key=None,
                device: str = None, local_models: Dict = None,
                api_models: List[str] = None, default_local_model: str = None,
                default_api_model: str = None):

        if api_key == None:
            local = True
                    
        if device == None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.local = local
        self.api_key = api_key
        self._model_device = device

        # Pick the correct model name
        if local:
            if model is None:
                model = default_local_model

            if type(model) == str or model is None:
                # Get from dictionary or use provided if not there
                model = local_models.get(model) or model

            if type(model) == str:
                self.model = load(model)
            elif hasattr(model, "model"):
                self.model = model
            else:
                self.model = model(device=device)
        else:
            if model is None or type(model) != str:
                model = default_api_model
    
            # All checks passed
            self.model = model

    def __call__(self):
        raise Exception("The base Task is not callable!")

    def finetune(self, *args, **kwargs):
        """
        Finetuning is not implemented for this task.
        """
        raise NotImplementedError("Finetuning is not implemented for this task")