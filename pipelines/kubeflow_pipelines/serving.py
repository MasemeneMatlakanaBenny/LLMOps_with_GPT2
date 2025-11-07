from kfp import dsl,compiler
from kfp.dsl import component,pipeline,Input,Output,Artifact

@component
def serving_artifacts(model_artifact:Output[Artifact],
                  tokenizer_artifact:Output[Artifact]):
    """A function meant for model serving in the workflow when we are creating this pipeline"""
    import mlflow
    import joblib
    import mlflow.pyfunc
    from src.configurations_mlflow import load_host,load_model_name,load_tokenizer_name
    from src.configurations_mlflow import set_mlflow_experiment,set_mlflow_tracking_uri
    from src.configurations import set_device

    model_name=load_model_name()
    tokenizer_name=load_tokenizer_name()

    stage="production"
    model_uri=f"models:/{model_name}/{stage}"
    tokenizer_uri=f"models:/{model_name}/{stage}"

    model=mlflow.pyfunc.load_model(model_uri=model_uri)
    tokenizer=mlflow.pyfunc.load_model(model_uri=model_uri)


    joblib.dump(model,model_artifact.path)
    joblib.dump(tokenizer,tokenizer_artifact.path)

@pipeline(
    name="serving-pipeline",
    pipeline_description="Pipeline for loading both the tokenizer and LLM"
)
def serving_pipeline():
    saved_arts=serving_artifacts()

    # now lets use the outputs and build the workflow using the created component:

compiler.Compiler().compile(
    pipeline_func=serving_pipeline,
    package_path="llm_pipelines/kubeflow_pipelines/serving_artifacts_pipeline.yaml"
)
