from configurations import load_tokenizer,load_model,set_device
from flask import Flask
from schemas import Prompt


##set the device first within the workflow:
device=set_device()


#load the model and the tokenizer
## load the model first:
model=load_model()

## load the tokenizer now:
tokenizer=load_tokenizer()
prompt=input("Enter the prompt here: ")

#validate the prompt

prompt_validation=Prompt(**prompt)

##create the application here:
app=Flask(__name__)

@app.route('/gpt2-response',method=['POST'])
# now if the prompt has been validated ,we should be able to generate the response:
def generated_text_by_model(input_text=prompt,model=model,tokenizer=tokenizer):
    """
    
    A function that will be used to generate the response from the input prompt
      that the user has entered.
      
      
      """

    inputs=tokenizer(input_text,return_tensors="pt") # return the tensors in pytorch format:

    # get the input ids + attention_mask:
    input_ids=inputs.input_ids.to(device)

    attention_mask=inputs.attention_mask.to(device)

    # now generate the text:

    generated_text=model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100
    )

    return tokenizer.decode(generated_text[0],skip_specials_tokens=True)

@app.route('/metrics',method=['POST'])
def evaluate_text_generation(generated_text):
    """This is a function that will be used to evaluate the generated text by the model"""

    import evaluate

    perplexity=evaluate.load("perplexity")

    results=perplexity.compute(
        model_id="gpt2",
        add_start_token=True,
        predictions=[generated_text]

    )

    return results
