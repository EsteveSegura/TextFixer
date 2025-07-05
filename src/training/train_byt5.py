from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import random
import json
import os

# 1. Create a tiny debug dataset with instructive prompts
DEBUG_DATA_PATH = "data/byt5_debug.jsonl"
debug_examples = [
    {"input": "Deobfuscate: H3ll0", "target": "Hello"},
    {"input": "Deobfuscate: W0rld", "target": "World"},
    {"input": "Deobfuscate: 7h15 15 4 t35t.", "target": "this is a test."},
    {"input": "Deobfuscate: 7h3 qu1ck br0wn f0x", "target": "the quick brown fox"},
    {"input": "Deobfuscate: 50m3 0bfu5c473d t3x7", "target": "some obfuscated text"},
    {"input": "Deobfuscate: 1337 5p34k", "target": "leet speak"},
    {"input": "Deobfuscate: 5up3r c00l", "target": "super cool"},
    {"input": "Deobfuscate: 7r41n m0d3l", "target": "train model"},
    {"input": "Deobfuscate: 7h15 15 fun", "target": "this is fun"},
    {"input": "Deobfuscate: 0p3n41", "target": "openai"},
]

# Save the debug dataset
os.makedirs(os.path.dirname(DEBUG_DATA_PATH), exist_ok=True)
with open(DEBUG_DATA_PATH, "w", encoding="utf-8") as f:
    for ex in debug_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Debug dataset saved to: {DEBUG_DATA_PATH}")

# 2. Load the debug dataset
raw_dataset = []
with open(DEBUG_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw_dataset.append(json.loads(line))

# Shuffle and split into train/validation (80/20 split for tiny set)
random.shuffle(raw_dataset)
split_idx = int(0.8 * len(raw_dataset))
train_data = raw_dataset[:split_idx]
val_data = raw_dataset[split_idx:]

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 3. Load tokenizer and model
print("Loading tokenizer and model: google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# 4. Tokenization function
def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing debug dataset...")
tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_val = val_dataset.map(preprocess, batched=True)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./byt5-finetuned-debug",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=2,
    logging_steps=1,
    save_steps=5,
    save_total_limit=1,
    num_train_epochs=10,
    fp16=True,  # Use float16 if your GPU supports it
    report_to="none",  # Remove if you use wandb or tensorboard
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# 7. Start training
print("Starting debug training...")
trainer.train()
print("Debug training completed!") 