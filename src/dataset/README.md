# TextFixer Dataset

This directory contains the dataset downloader for the TextFixer LLM model.

## Overview

The dataset downloader downloads a subset of the [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) from HuggingFace. FineWeb is a large-scale, high-quality text dataset derived from CommonCrawl web data.

## Features

- Downloads 3000 samples from the FineWeb dataset
- Processes and saves data in multiple formats (JSON and TXT)
- Provides detailed statistics about the downloaded dataset
- Uses streaming to handle large datasets efficiently
- Includes metadata like language scores and token counts

## Usage

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Downloader

```bash
python src/dataset/main.py
```

### Configuration

You can modify the following parameters in `main.py`:

- `num_samples`: Number of samples to download (default: 3000)
- `output_dir`: Directory to save the processed dataset (default: "data")

## Output Files

The script generates the following files in the `data/` directory:

- `fineweb_subset.json`: Complete dataset with metadata in JSON format
- `fineweb_subset.txt`: Plain text file with one sample per line

## Dataset Structure

Each sample in the JSON file contains:

```json
{
  "id": 0,
  "text": "Sample text content...",
  "language": "en",
  "language_score": 0.95,
  "token_count": 150
}
```

## Dataset Statistics

The script provides statistics including:
- Total number of samples
- Total and average token counts
- Total and average character counts

## FineWeb Dataset Information

- **Source**: CommonCrawl web data (2013-2024)
- **Language**: English (filtered with language score ≥ 0.65)
- **Quality**: High-quality text with filtering for:
  - Repetition and quality metrics
  - NSFW content removal
  - PII anonymization
  - Deduplication

## License

The FineWeb dataset is released under the Open Data Commons Attribution License (ODC-By) v1.0. 