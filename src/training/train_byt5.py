from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, TrainerCallback
import random
import json
import os
import torch

# 1. Load the full dataset (JSONL format with 'input' and 'target' fields)
DATA_PATH = "data/byt5_dataset.jsonl"

print(f"Loading dataset from: {DATA_PATH}")
raw_dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw_dataset.append(json.loads(line))

# Shuffle and split into train/validation (90/10 split)
random.shuffle(raw_dataset)
split_idx = int(0.9 * len(raw_dataset))
train_data = raw_dataset[:split_idx]
val_data = raw_dataset[split_idx:]

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 2. Load tokenizer and model
print("Loading tokenizer and model: google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# 3. Tokenization function
def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_val = val_dataset.map(preprocess, batched=True)

# 4. Training arguments (tuned for ~6k examples)
training_args = TrainingArguments(
    output_dir="./byt5-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=250,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=5,
    fp16=True,  # Use float16 if your GPU supports it
    report_to="none",  # Remove if you use wandb or tensorboard
)

# 5. Inference callback for periodic inference during training
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, example_input, steps=500):
        self.tokenizer = tokenizer
        self.example_input = example_input
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()
            device = next(model.parameters()).device
            inputs = self.tokenizer(self.example_input, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Inference for: {self.example_input}")
            print(f"Output: {output}\n")
            model.train()

# Choose a sample input from the validation set for inference
if len(val_data) > 0:
    example_input = val_data[0]["input"]
else:
    example_input = "Deobfuscate: H3ll0 w0rld"

inference_callback = InferenceCallback(tokenizer, example_input, steps=500)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    callbacks=[inference_callback],
)

# 7. Start training
print("Starting training...")
trainer.train()
print("Training completed!") 