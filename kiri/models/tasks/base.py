from typing import Dict, List
import logging

logger = logging.getLogger("info")

class Task:
    def __init__(self, model, local=False, api_key=None,
                device: str = "cpu", init=False, local_models: Dict = None,
                api_models: List[str] = None, default_local_model: str = None,
                default_api_model: str = None):
        self.local = local
        self.api_key = api_key
        self.device = device
        self.init = init

        # Pick the correct model name
        if local:
            if model is None:
                model = default_local_model

            if type(model) == str or model is None:
                # Get from dictionary or use provided if not there
                model = local_models.get(model) or model
                if model not in local_models.values():
                    logger.warning(f"Model '{model}' is not officially supported, but it may function.")
                
            self.model = model
        else:
            if model is None or type(model) != str:
                model = default_api_model

            if api_key is None:
                raise ValueError(
                    "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")
            
            if model not in api_models:
                raise ValueError(f"Model '{model}' is not supported for this task. Please use one of: {', '.join(api_models)}")
    
            # All checks passed
            self.model = model

    def __call__(self):
        raise Exception("The base Task is not callable!")