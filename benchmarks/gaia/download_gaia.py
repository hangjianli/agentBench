import os
from datasets import load_dataset
import pandas as pd

def main():
    print("Downloading GAIA dataset from Hugging Face...")
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the GAIA dataset (using 2023_all config)
        dataset = load_dataset("GAIA-benchmark/GAIA", "2023_all")
        
        for split in dataset.keys():
            print(f"Saving split: {split}")
            df = dataset[split].to_pandas()
            output_file = os.path.join(output_dir, f"{split}.jsonl")
            df.to_json(output_file, orient="records", lines=True)
            print(f"Saved {len(df)} records to {output_file}")
            
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    main()
