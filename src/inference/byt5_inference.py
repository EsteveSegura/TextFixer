import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Path to the directory where the fine-tuned ByT5 model checkpoints are saved
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../byt5-finetuned')

# Find the checkpoint with the highest number
checkpoint_dirs = [d for d in os.listdir(BASE_MODEL_DIR) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
if not checkpoint_dirs:
    raise FileNotFoundError(f"No checkpoint directories found in {BASE_MODEL_DIR}")

# Extract checkpoint numbers and find the highest
checkpoint_numbers = [int(d.split('-')[-1]) for d in checkpoint_dirs]
max_checkpoint = max(checkpoint_numbers)
highest_checkpoint_dir = os.path.join(BASE_MODEL_DIR, f'checkpoint-{max_checkpoint}')

print(f"Loading tokenizer and model from: {highest_checkpoint_dir}")
tokenizer = AutoTokenizer.from_pretrained(highest_checkpoint_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(highest_checkpoint_dir)

def main():
    # Example instructive obfuscated input for inference
    obfuscated_input = "H3ll0 w0rld"
    instructive_input = f"Deobfuscate: {obfuscated_input}"
    print(f"Instructive input: {instructive_input}")

    # Tokenize the input
    inputs = tokenizer(instructive_input, return_tensors="pt")

    # Generate output (move to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate prediction
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Model output:")
    print(output)

if __name__ == "__main__":
    main() 