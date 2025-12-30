"""
Download LLaVA-Instruct-150K dataset manually
"""
import json
import os
import requests
from tqdm import tqdm


def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def main():
    # Create data directory
    os.makedirs("data/llava_instruct_150k", exist_ok=True)
    
    # Download the JSON file
    url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
    output_path = "data/llava_instruct_150k/llava_instruct_150k.json"
    
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        
        # Verify it's valid JSON
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            print(f"Valid JSON file with {len(data)} examples")
            return
        except Exception as e:
            print(f"Invalid JSON file, re-downloading: {e}")
            os.remove(output_path)
    
    print(f"Downloading {url}...")
    download_file(url, output_path)
    
    # Verify the download
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        print(f"\nSuccessfully downloaded {len(data)} examples")
        
        # Print sample
        print("\nSample entry:")
        print(json.dumps(data[0], indent=2))
        
    except Exception as e:
        print(f"\nError loading JSON: {e}")


if __name__ == "__main__":
    main()