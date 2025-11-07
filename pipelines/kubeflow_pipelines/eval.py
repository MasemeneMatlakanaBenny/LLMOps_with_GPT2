from kfp import dsl,compiler
from kfp.dsl import pipeline,component
from text_generation import text_generation_pipeline

@component
def model_evaluation():
    """A function meant for model serving in the workflow when we are creating this pipeline"""
    import evaluate
    
    # the text generation pipeline function will return the response from the model:
    response_by_model=text_generation_pipeline()

    
    # lets use perplexity to evaluate the model's responses
    perplexity=evaluate("perplexity")

    metrics=perplexity.compute(
        model_id="gpt2",
        add_start_token=True,
        predictions=response_by_model
    )

    return metrics

@pipeline(
    name="llm-evaluation-pipeline",
    description="Pipeline that will be used to evaluate the generated responses by the LLM"
)
def evaluation_pipeline():
    results= model_evaluation()

    return results

compiler.Compiler().compile(
    pipeline_func=evaluation_pipeline,
    package_path="llm_pipelines/kubeflow_pipelines/eval.yaml"
)


   
   
