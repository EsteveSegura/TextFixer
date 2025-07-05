#!/usr/bin/env python3
"""
Simple script to add leetspeak obfuscation to the existing dataset.
"""

import json
import random
import re

# Leetspeak mappings
LEET_MAPPINGS = {
    "a": [
        "4",
        "/\\",
        "@",
        "/-\\",
        "^",
        "aye",
        "(L",
        "Д"
    ],
    "b": [
        "I3",
        "8",
        "13",
        "|3",
        "ß",
        "!3",
        "(3",
        "/3",
        ")3",
        "|-]",
        "j3",
        "6"
    ],
    "c": [
        "[",
        "¢",
        "{",
        "<",
        "(",
        "©"
    ],
    "d": [
        ")",
        "|)",
        "(|",
        "[)",
        "I>",
        "|>",
        "?",
        "T)",
        "I7",
        "cl",
        "|}",
        ">",
        "|]"
    ],
    "e": [
        "3",
        "&",
        "£",
        "€",
        "ë",
        "[-",
        "|=-"
    ],
    "f": [
        "|=",
        "ƒ",
        "|#",
        "ph",
        "/=",
        "v"
    ],
    "g": [
        "&",
        "6",
        "(_+",
        "9",
        "C-",
        "gee",
        "(?,",
        "[,",
        "{,",
        "<-",
        "(."
    ],
    "h": [
        "#",
        "/-/",
        "[-]",
        "]-[",
        ")-(",
        "(-)",
        ":-:",
        "|~|",
        "|-|",
        "]~[",
        "}{",
        "!-!",
        "1-1",
        "\\\\-/",
        "I+I",
        "/\\-\\"
    ],
    "i": [
        "1",
        "[]",
        "|",
        "!",
        "eye",
        "3y3",
        "]["
    ],
    "j": [
        ",_|",
        "_|",
        "._|",
        "._]",
        "_]",
        ",_]",
        "]",
        ";",
        "1"
    ],
    "k": [
        ">|",
        "|<",
        "/<",
        "1<",
        "|c",
        "|(",
        "|{"
    ],
    "l": [
        "1",
        "£",
        "7",
        "|_",
        "|"
    ],
    "m": [
        "/\\/\\",
        "/V\\",
        "JVI",
        "[V]",
        "[]V[]",
        "|\\/|",
        "^^",
        "<\\/>",
        "{V}",
        "(v)",
        "(V)",
        "|V|",
        "nn",
        "IVI",
        "|\\|\\\\",
        "]\\/[",
        "1^1",
        "ITI",
        "JTI"
    ],
    "n": [
        "^/",
        "|\\|",
        "/\\/",
        "[\\]",
        "<\\>",
        "{\\}",
        "|V",
        "/V",
        "И",
        "^",
        "ท"
    ],
    "o": [
        "0",
        "Q",
        "()",
        "oh",
        "[]",
        "p",
        "<>",
        "Ø"
    ],
    "p": [
        "|*",
        "|o",
        "|º",
        "?",
        "|^",
        "|>",
        "|\"",
        "9",
        "[]D",
        "|°",
        "|7"
    ],
    "q": [
        "(_,)",
        "9",
        "()_",
        "2",
        "0_",
        "<|",
        "&"
    ],
    "r": [
        "I2",
        "|`",
        "|~",
        "|?",
        "/2",
        "|^",
        "lz",
        "|9",
        "2",
        "12",
        "®",
        "[z",
        "Я",
        ".-",
        "|2",
        "|-"
    ],
    "s": [
        "5",
        "$",
        "z",
        "§",
        "ehs",
        "es",
        "2"
    ],
    "t": [
        "7",
        "+",
        "-|-",
        "']['",
        "†",
        "\"|\"",
        "~|~"
    ],
    "u": [
        "(_)",
        "|_|",
        "v",
        "L|",
        "µ",
        "บ"
    ],
    "v": [
        "\\/",
        "|/",
        "\\\\|"
    ],
    "w": [
        "\\/\\/",
        "VV",
        "\\\\N",
        "'//",
        "\\\\\\\\'",
        "\\\\^/",
        "(n)",
        "\\\\V/",
        "\\\\X/",
        "\\\\|/",
        "\\\\_|_/",
        "\\\\_:_/",
        "Ш",
        "Щ",
        "uu",
        "2u",
        "\\\\\\\\//\\\\\\\\//",
        "พ",
        "v²"
    ],
    "x": [
        "><",
        "Ж",
        "}{",
        "ecks",
        "×",
        "?",
        ")(",
        "]["
    ],
    "y": [
        "j",
        "`/",
        "Ч",
        "7",
        "\\\\|/",
        "¥",
        "\\\\//"
    ],
    "z": [
        "2",
        "7_",
        "-/_",
        "%",
        ">_",
        "s",
        "~/_",
        "-\\\\_",
        "-|_"
    ]
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