from kfp import dsl,compiler
from kfp.dsl import component,pipeline,Input,Output,Artifact
from serving import serving_artifacts

@component
def generating_text_component(text:str):
    """A function meant for model serving in the workflow when we are creating this pipeline"""
   
    import joblib
    saved_arts=serving_artifacts()

    model=saved_arts.outputs['model_artifact']

    tokenizer=saved_arts.outputs['tokenizer_artifact']

    ## now lets generate some text within the component:
    from src.chat_with_model import generated_text_by_model

    response_model=generated_text_by_model(input_text=text,model=model,tokenizer=tokenizer)

    return response_model

@pipeline(
    name="text-generation-pipeline",
    description="Text generation pipeline"
)
def text_generation_pipeline():
    text_generated=generating_text_component("")

compiler.Compiler().compile(
    pipeline_func=text_generation_pipeline,
    package_path="llm_pipelines/kubeflow_pipelines/text_generation_pipeline.yaml"
)