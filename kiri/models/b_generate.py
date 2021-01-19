from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

model = None
tokenizer = None
m_type = None

model_types = {
    't5': T5ForConditionalGeneration,
    'gpt2': GPT2LMHeadModel
}

tokenizer_types = {
    't5': T5Tokenizer,
    'gpt2': GPT2Tokenizer
}

instance_names = {
    't5': 't5-small',
    'gpt2': 'gpt2'
}

def b_generate(input_text: str, model_type: str = 'gpt2',
               model_name: str = None, tokenizer_name: str = None,
               local: bool = True, device: str = 'cpu'):

    global model
    global tokenizer
    global m_type
    
    if local:

        if model_type not in model_types.keys():
            raise ValueError(f'Model type not supported! Supported types: {list(model_types.keys())}')
        
        if not model or model_type != m_type:
            if not model_name:
                model = model_types[model_type].from_pretrained(instance_names[model_type])
            else:
                model = model_types[model_type].from_pretrained(model_name)
            model.to(device)
        
        if not tokenizer or model_type != m_type:
            if not tokenizer_name:
                tokenizer = tokenizer_types[model_type].from_pretrained(instance_names[model_type])
            else:
                model = model_types[model_type].from_pretrained(tokenizer_name)

        m_type = model_type

        encoded_input = tokenizer.encode(input_text, return_tensors='pt')
        encoded_input = encoded_input.to(device)

        output = model.generate(input_ids=encoded_input, num_return_sequences=1)

        output = tokenizer.decode(output[0])
        return output
    else:
        raise ValueError('Non-local inference is not currently implemented')






