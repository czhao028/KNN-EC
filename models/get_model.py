from transformers import AutoTokenizer, AutoModel

def get_model(config):
    model_name=config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model,tokenizer
    
    