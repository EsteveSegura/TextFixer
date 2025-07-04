import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the directory where the trained model checkpoints are saved
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models/textfixer')

# Find the checkpoint with the highest number
checkpoint_dirs = [d for d in os.listdir(BASE_MODEL_DIR) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
if not checkpoint_dirs:
    raise FileNotFoundError(f"No checkpoint directories found in {BASE_MODEL_DIR}")

# Extract checkpoint numbers and find the highest
checkpoint_numbers = [int(d.split('-')[-1]) for d in checkpoint_dirs]
max_checkpoint = max(checkpoint_numbers)
highest_checkpoint_dir = os.path.join(BASE_MODEL_DIR, f'checkpoint-{max_checkpoint}')

print(f"Loading model and tokenizer from: {highest_checkpoint_dir}")
tokenizer = AutoTokenizer.from_pretrained(highest_checkpoint_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(highest_checkpoint_dir, trust_remote_code=True)

# Example prompt for inference
def main():
    prompt = "This is an example of"
    print(f"Prompt: {prompt}")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output from the model
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode and print the generated text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:")
    print(result)

if __name__ == "__main__":
    main() 