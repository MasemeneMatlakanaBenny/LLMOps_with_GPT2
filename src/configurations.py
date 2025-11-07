import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM


## load the device first:
def set_device()->torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

## now lets create a function that will load the model:
def load_model(model_name:str=transformers.AutoModelForCausalLM,device:torch.device=set_device())-> transformers.AutoModelForCausalLM:
    """This is a function that will be called to load the model directly for reusability purposes"""

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto"
    )

    model.config.pad_token_id=model.config.eos_token_id

    return model.to(device)

## now lets create a function that will load the tokenizer:
def load_tokenizer(model_name:str=transformers.AutoModelForCausalLM)->transformers.AutoModelForCausalLM:
    """This is a function that will be called to load the tokenizer directly for reusability"""

    tokenizer=AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    tokenizer.pad_token=tokenizer.eos_token

    return tokenizer

## generate text