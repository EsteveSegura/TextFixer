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
    # Tokenize input and target in one call using text_target (new API)
    model_inputs = tokenizer(
        example["input"],
        text_target=example["target"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
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
    learning_rate=2e-4,
    warmup_steps=500,
    fp16=False,
    max_grad_norm=1.0,
    report_to="none",
)

# 5. Inference callback for periodic inference during training
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, validation_data, steps=500):
        self.tokenizer = tokenizer
        self.validation_data = validation_data
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            model.eval()
            device = next(model.parameters()).device
            
            # Select a random example from validation data
            if self.validation_data:
                random_example = random.choice(self.validation_data)
                example_input = random_example["input"]
            else:
                example_input = "Deobfuscate: H3ll0 w0rld"
            
            inputs = self.tokenizer(example_input, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Inference for: {example_input}")
            print(f"Output: {output}\n")
            model.train()

inference_callback = InferenceCallback(tokenizer, val_data, steps=500)

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