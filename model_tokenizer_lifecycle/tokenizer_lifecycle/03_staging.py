import mlflow
from src.configurations_mlflow import load_host,load_tokenizer_name
from src.configurations_mlflow import set_mlflow_experiment,set_mlflow_tracking_uri,mlflow_client
from src.configurations import set_device,load_model

# use the imported functions to create variables for model registry workflow:
## create the device first-> gpu/cuda or cpu

set_mlflow_experiment()
set_mlflow_tracking_uri()
device=set_device()


tokenizer_name=load_tokenizer_name()
model_version="1"

client=mlflow_client()

client.transition_model_version_stage(
    name=tokenizer_name,
    version=model_version,
    stage="staging"
)