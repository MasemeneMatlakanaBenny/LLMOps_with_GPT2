from src.configurations_mlflow import test_model_registry,load_model_name

#get the model parameters first:
model_name=load_model_name()

model_version="1"

test_model_registry(name=model_name,version=model_version)
