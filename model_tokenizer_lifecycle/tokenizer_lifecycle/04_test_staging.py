from src.configurations_mlflow import test_model_versioning,load_tokenizer_name

#get the model parameters first:
model_name=load_tokenizer_name()

model_stage="staging"

test_model_versioning(name=model_name,stage=model_stage)