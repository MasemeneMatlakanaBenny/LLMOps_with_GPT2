##this is where we store all the variables for mlflow workflow:
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException


host="httpp://127.0.0.1:5000"
exp_name="LLMOps_with_GPT2_Model"
exp_description="Getting started with LLMOps for GPT2 Model and evaluating prompts with Perplexity metric"
tags={
    "project_name":"LLMOps with GPT Model",
    "key metrics":"Perplexity",
    "team":"ML/AI team",
    "team lead":"Masemene Matlakana Benny",
    "mlflow.note.content":exp_description
}


# this function will directly load the local host or server used on the Virtual Machine for configuring mlflow
def load_host():
    return host

# create a function that we can use to 

## now we are able to load the tags for the experiment created within the mlflow workflow for this project:
def load_tags():
    return tags

# used to load both the model and tokenizer names but the ones we are using to register them:
def load_model_name():
    return "gpt2_model"

def load_tokenizer_name():
    return "gpt2_tokenizer"

## now lets use the MlflowClient :
def mlflow_client(host=host):
    client=MlflowClient(tracking_uri=host)

    return client

## set the experiment within the workflow :
def set_mlflow_tracking_uri(host=host):
    return mlflow.set_tracking_uri(host)

def set_mlflow_experiment(name=exp_name,host=host):
    return mlflow.set_experiment(experiment_name=name)

def test_model_registry(name, version):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, version=version)
        print(f"Model '{name}' version {version} exists")
    except RestException:
        print(f" Model '{name}' version {version} not found")

def test_model_versioning(name, stage):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, stage=stage)
        print(f"Model '{name}' at {stage} exists")
    except RestException:
        print(f"Model '{name}' at {stage} not found")
