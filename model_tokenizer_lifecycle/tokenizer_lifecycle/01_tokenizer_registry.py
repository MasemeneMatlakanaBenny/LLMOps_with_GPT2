import mlflow
from src.configurations_mlflow import load_host,load_tokenizer_name
from src.configurations_mlflow import set_mlflow_experiment,set_mlflow_tracking_uri
from src.configurations import set_device,load_tokenizer

# use the imported functions to create variables for model registry workflow:
## create the device first-> gpu/cuda or cpu
device=set_device()

## load the model directly from huggingface
tokenizer=load_tokenizer()

## set the experiment and tracking uri within mlflow:
host=load_host()

tokenizer_name=load_tokenizer_name()

set_mlflow_experiment()
set_mlflow_tracking_uri()

# create the run name that will be used to register the model:
run_name="gpt2_run"
with mlflow.start_run(run_name=run_name) as run:
    mlflow.pyfunc.log_model(tokenizer,registered_model_name=tokenizer_name)
