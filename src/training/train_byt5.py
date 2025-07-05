from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import random

# 1. Load the dataset (JSONL format with 'input' and 'target' fields)
DATA_PATH = "data/byt5_dataset.jsonl"

print(f"Loading dataset from: {DATA_PATH}")
raw_dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw_dataset.append(eval(line))  # Each line is a dict

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

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./byt5-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=True,  # Use float16 if your GPU supports it
    report_to="none",  # Remove if you use wandb or tensorboard
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# 6. Start training
print("Starting training...")
trainer.train()
print("Training completed!") 