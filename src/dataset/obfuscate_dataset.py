#!/usr/bin/env python3
"""
Simple script to add leetspeak obfuscation to the existing dataset.
"""

import json
import random
import re

# Leetspeak mappings
LEET_MAPPINGS = {
    'a': ['4', '@', 'A'],
    'e': ['3', 'E'],
    'i': ['1', '!', 'I'],
    'o': ['0', 'O'],
    's': ['5', '$', 'S'],
    't': ['7', 'T'],
    'l': ['1', '|', 'L'],
    'g': ['6', 'G'],
    'b': ['8', 'B'],
    'z': ['2', 'Z']
}

def convert_word_to_leetspeak(word, char_percentage):
    """Convert a word to leetspeak with given percentage of characters."""
    if not word:
        return word
    
    # Find convertible characters
    convertible_chars = []
    for i, char in enumerate(word.lower()):
        if char in LEET_MAPPINGS:
            convertible_chars.append(i)
    
    if not convertible_chars:
        return word
    
    # Calculate how many characters to convert
    num_chars_to_convert = max(1, int(len(convertible_chars) * char_percentage / 100))
    chars_to_convert = random.sample(convertible_chars, min(num_chars_to_convert, len(convertible_chars)))
    
    # Convert the word
    word_list = list(word)
    for char_idx in chars_to_convert:
        original_char = word_list[char_idx].lower()
        if original_char in LEET_MAPPINGS:
            replacement = random.choice(LEET_MAPPINGS[original_char])
            word_list[char_idx] = replacement
    
    return ''.join(word_list)

def obfuscate_text(text):
    """Convert text to leetspeak with random percentages."""
    words = re.findall(r'\b\w+\b', text)
    
    if not words:
        return text
    
    # Random word selection percentage (25%, 50%, 75%, 100%)
    word_percentage = random.choice([25, 50, 75, 100])
    num_words_to_convert = max(1, int(len(words) * word_percentage / 100))
    
    # Randomly select words to convert
    words_to_convert = random.sample(range(len(words)), min(num_words_to_convert, len(words)))
    
    # Convert selected words
    converted_words = []
    for i, word in enumerate(words):
        if i in words_to_convert:
            # Random character conversion percentage (25%, 50%, 100%)
            char_percentage = random.choice([25, 50, 100])
            converted_word = convert_word_to_leetspeak(word, char_percentage)
            converted_words.append(converted_word)
        else:
            converted_words.append(word)
    
    # Reconstruct text
    result = text
    for orig_word, conv_word in zip(words, converted_words):
        if orig_word != conv_word:
            pattern = r'\b' + re.escape(orig_word) + r'\b'
            result = re.sub(pattern, conv_word, result)
    
    return result

def main():
    # Load the original dataset
    with open('data/fineweb_subset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Add obfuscated text to each sample
    for sample in dataset:
        original_text = sample.get('text', '')
        sample['text_obfuscated'] = obfuscate_text(original_text)
    
    # Save the new dataset
    with open('data/fineweb_subset_obfuscated.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Dataset obfuscated successfully!")

if __name__ == "__main__":
    main() 