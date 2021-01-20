from transformers import AutoModelForPreTraining, AutoTokenizer

DEFAULT = "gpt2"

model = None
tokenizer = None
m_checkpoint = None
t_checkpoint = None

def generate(input_text: str, model_name: str = None, 
               tokenizer_name: str = None, local: bool = True, 
               device: str = 'cpu'):

    global model
    global tokenizer
    global m_checkpoint
    global t_checkpoint
    
    if local:
        # Initialise model, allow switching if new one is provided.
        if not model or (model_name and model_name != m_checkpoint):
            if not model_name:
                model = AutoModelForPreTraining.from_pretrained(DEFAULT)
            else:
                model = AutoModelForPreTraining.from_pretrained(model_name)
            model.to(device)
        
        # Initialise tokenizer, allow switching if new one is provided.
        if not tokenizer or (tokenizer_name and tokenizer_name != t_checkpoint):
            if not tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        
        # Set the checkpoints used to load the model/tokenizer
        m_checkpoint = model_name
        t_checkpoint = tokenizer_name

        is_list = isinstance(input_text, list)

        encoded_input = tokenizer.encode(input_text, return_tensors='pt')
        encoded_input = encoded_input.to(device)

        output = model.generate(input_ids=encoded_input, num_return_sequences=1)

        if is_list:
            return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output]
        else:
            return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        raise ValueError('Non-local inference is not currently implemented')






