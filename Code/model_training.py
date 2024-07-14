import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Load the dataset
df = pd.read_csv('large_indian_recipes.csv')

# Combine the ingredients and instructions into a single text column
df['text'] = 'Title: ' + df['title'] + '\nIngredients: ' + df['ingredients'] + '\nInstructions: ' + df['instructions']

# Extract the text column
recipes = df['text'].tolist()

# Preprocess the text data
def preprocess_recipe(recipe):
    return recipe.lower()

recipes = [preprocess_recipe(recipe) for recipe in recipes]

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the text data with padding
tokenized_recipes = tokenizer(recipes, return_tensors='pt', truncation=True, padding=True, max_length=512)

# Create a dataset object
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

dataset = RecipeDataset(tokenized_recipes)

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs 3
    per_device_train_batch_size=2,   # batch size per device during training
    save_steps=10_00,               # number of updates steps before checkpoint saves 10_000
    save_total_limit=2,              # limit the total amount of checkpoints. Deletes the older checkpoints
    logging_dir='./logs',            # directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

