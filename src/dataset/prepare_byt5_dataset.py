import json

"""
Script to convert the obfuscated dataset to JSONL format for ByT5 fine-tuning.
"""

def convert_to_byt5_jsonl():
    """Convert the obfuscated dataset to JSONL format for ByT5 with an instructive input."""
    # Load the obfuscated dataset
    with open('data/fineweb_subset_obfuscated.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Create JSONL file for ByT5
    with open('data/byt5_dataset.jsonl', 'w', encoding='utf-8') as f:
        count = 0
        for sample in dataset:
            text_obfuscated = sample.get('text_obfuscated', '')
            text_original = sample.get('text', '')
            if not text_obfuscated or not text_original:
                continue
            # Use an instructive prompt for the input
            input_text = f"Deobfuscate: {text_obfuscated}"
            json_line = {"input": input_text, "target": text_original}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            count += 1
    print(f"ByT5 dataset converted to JSONL format with instructive input: data/byt5_dataset.jsonl")
    print(f"Total samples: {count}")

if __name__ == "__main__":
    convert_to_byt5_jsonl() 