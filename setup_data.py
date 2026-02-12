
import os
import zipfile
import shutil
import re
from tqdm import tqdm

# Configuration
SOURCE_DIR = r'D:\fullSADT'  # Path to zip files
DEST_DIR = r'd:\DSADNet-main\data\raw' # Path to extract to (data/raw)

def setup_data():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")

    # List all zip files
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.zip')]
    print(f"Found {len(files)} zip files in {SOURCE_DIR}")

    for zip_file in tqdm(files, desc="Processing datasets"):
        # Parse subject ID from filename (e.g. s01_051017m.set.zip -> s01)
        # Using simple split by '_'
        try:
            subject_id = zip_file.split('_')[0]
        except IndexError:
            print(f"Skipping {zip_file}: Cannot parse subject ID.")
            continue

        subject_dir = os.path.join(DEST_DIR, subject_id)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        
        zip_path = os.path.join(SOURCE_DIR, zip_file)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                
                # We need to extract .set and .fdt files, flattening the structure
                for member in file_list:
                    filename = os.path.basename(member)
                    # Skip directories
                    if not filename:
                        continue
                        
                    if filename.endswith('.set') or filename.endswith('.fdt'):
                        # Construct target path
                        target_path = os.path.join(subject_dir, filename)
                        
                        # Read content and write to target
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                            
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is a bad zip file.")
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")

    print("Data setup complete.")

if __name__ == "__main__":
    setup_data()
