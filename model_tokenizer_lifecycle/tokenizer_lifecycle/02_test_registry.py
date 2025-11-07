from src.configurations_mlflow import test_model_registry,load_tokenizer_name

#get the model parameters first:
tokenizer_name=load_tokenizer_name()

tokenizer_version="1"

test_model_registry(name=tokenizer_name,version=tokenizer_version)