import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model_name = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate recipes
def generate_recipe(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create the Gradio UI
def get_recipe(prompt):
    return generate_recipe(prompt)

iface = gr.Interface(
    fn=get_recipe,
    inputs=gr.Textbox(lines=2, placeholder="Enter ingredients or a partial recipe..."),
    outputs="text",
    title="Indian Recipe Generator",
    description="Enter a prompt to generate an Indian recipe"
)

iface.launch()
