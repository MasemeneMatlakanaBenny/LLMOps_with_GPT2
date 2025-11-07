from prefect import task,flow


prompt=input("Enter the prompt here: ")

@task
def model_serving_task():
    """A component of the workflow that will be used for LLM serving purposes"""

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

    return model,tokenizer

@task
def text_generation_task(model,tokenizer,prompt=prompt):

    """A component of the workflow or pipeline that will be used for text generation"""
    # rewrite the entire code here again 
    ## or simply just load the chat_with_model library that we create earlier on 
    ## option 2 works much better -> reusability purposes in the ml workflow
    from src.chat_with_model import generated_text_by_model

    response_model=generated_text_by_model(input_text=prompt,model=model,tokenizer=tokenizer)

    return response_model


@task
def model_evaluation_task(response):
    """A component of the workflow or pipeline that will be used for model evaluation"""
    import evaluate
    
    # lets use perplexity to evaluate the model's responses
    perplexity=evaluate("perplexity")

    metrics=perplexity.compute(
        model_id="gpt2",
        add_start_token=True,
        predictions=response
    )

    return metrics


@task
def save_model_metrics(metrics):
    import joblib
    joblib.dump(metrics,"src/metrics.pkl")

@flow
def pipeline_automation():
    model,tokenizer=model_serving_task()
    model_response=text_generation_task(model=model,tokenizer=tokenizer)
    metrics_model_response=model_evaluation_task(response=model_response)

    save_model_metrics(metrics=metrics_model_response)

if __name__=="__main__":
    pipeline_automation()
