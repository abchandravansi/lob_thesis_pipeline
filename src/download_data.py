import os
import zipfile
import requests

def download_and_extract_fi2010(target_dir="data"):
    # Official direct API package payload endpoint for the FI-2010 package
    dataset_url = "https://fairdata.fi"
    
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "fi2010.zip")
    
    # 1. Download tracking sequence
    if not os.path.exists(zip_path):
        print("Downloading FI-2010 dataset from official mirror...")
        try:
            # Set browser user-agent headers to bypass firewall bot blockers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Using requests.get handles modern SSL/TLS context natively on Windows
            with requests.get(dataset_url, headers=headers, stream=True, timeout=60) as r:
                # Catch server-side HTTP errors (e.g., 403, 404, 500) immediately
                r.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024): # Stream in 1MB blocks
                        if chunk:
                            f.write(chunk)
            print("Download completed successfully.")
        except Exception as e:
            print(f"An error occurred during download: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path) # Clean up partial broken files
            return
    else:
        print("Dataset zip already exists. Skipping network download.")

    # 2. Extraction tracking sequence
    print("Unpacking zip file contents...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Extraction successful! Files are stored inside: {os.path.abspath(target_dir)}")
        
        # Housekeeping: Remove the zip archive file to save local disk storage space
        os.remove(zip_path)
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        print("\n[Tip] The server might have returned an invalid HTML error document instead of the zip file.")
        print("Try deleting the data/ folder completely and running this script again.")

if __name__ == "__main__":
    # Adjust directory tracking depth dynamically depending on current run execution paths
    if os.path.basename(os.getcwd()) == "src":
        download_and_extract_fi2010(target_dir="../data")
    else:
        download_and_extract_fi2010(target_dir="data")
