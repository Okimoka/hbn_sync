import os
import zipfile
import shutil
import sys
from pathlib import Path

"""
This unpacks the zips downloaded using dl_nemar.py
It assumes the default folder name (nemar_zips).
The staging area again avoids artifacts/incomplete results.
Final results will be in nemar_zips_unpacked.
When the script aborts and is restarted, it will also clear the staging area fully before continuing.

LLM Code
"""

# Configuration
SOURCE_DIR = Path("./nemar_zips")
STAGING_DIR = Path("./nemar_zips_unpacked.staging")
DEST_DIR = Path("./nemar_zips_unpacked")

def ensure_dirs():
    """Creates necessary directories if they don't exist."""
    for d in [SOURCE_DIR, STAGING_DIR, DEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Directories checked:\n- Source: {SOURCE_DIR}\n- Staging: {STAGING_DIR}\n- Dest: {DEST_DIR}\n")

def get_root_folder_name(zip_ref):
    """
    Analyzes the zip file to determine the single root folder name.
    Returns the root folder name or None if structure is invalid.
    """
    root_dirs = set()
    
    # We only look at the first component of the path
    for name in zip_ref.namelist():
        # Clean path separators
        clean_name = name.replace('\\', '/')
        
        # Split path parts
        parts = clean_name.split('/')
        top_level = parts[0]
        
        # Skip  empty strings
        if not top_level:
            continue
            
        root_dirs.add(top_level)

    if len(root_dirs) == 1:
        return list(root_dirs)[0]
    else:
        print(f"Error: Zip structure invalid. Found multiple or no root folders: {root_dirs}")
        return None

def process_zips():
    ensure_dirs()
    
    # Get list of zip files
    zip_files = sorted(list(SOURCE_DIR.glob("*.zip")))
    
    if not zip_files:
        print("No .zip files found in source directory.")
        return

    print(f"Found {len(zip_files)} zip files to process.")

    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n[{i}/{len(zip_files)}] Processing: {zip_path.name}...")
        
        # Clear staging area strictly before starting to avoid collisions
        if any(STAGING_DIR.iterdir()):
            print("Cleaning staging area...")
            for item in STAGING_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 1. Verify structure inside zip (fast, reads header only)
                root_folder_name = get_root_folder_name(zf)
                if not root_folder_name:
                    print(f"SKIPPING {zip_path.name}: Could not identify a single root folder.")
                    continue

                final_dest_path = DEST_DIR / root_folder_name
                if final_dest_path.exists():
                    print(f"SKIPPING {zip_path.name}: Destination already exists ({final_dest_path}).")
                    continue

                # 2. Extract to Staging
                print(f"  Extracting to staging (this may take a while)...")
                # Note: zipfile handles Zip64 (files >4GB) automatically in Python 3
                zf.extractall(path=STAGING_DIR)

                # 3. Simple Verification
                expected_staging_path = STAGING_DIR / root_folder_name
                if not expected_staging_path.exists():
                    print(f"ERROR: Extraction finished but folder {root_folder_name} not found in staging.")
                    continue
                
                print("  Verification successful: Root folder found in staging.")

                # 4. Move to Destination
                print(f"  Moving to {DEST_DIR}...")
                shutil.move(str(expected_staging_path), str(DEST_DIR))
                
                print(f"SUCCESS: Finished {zip_path.name}")

        except zipfile.BadZipFile:
            print(f"ERROR: {zip_path.name} is a corrupted zip file.")
        except Exception as e:
            print(f"CRITICAL ERROR processing {zip_path.name}: {e}")

    # Final cleanup of staging
    try:
        if STAGING_DIR.exists():
            STAGING_DIR.rmdir() # Only removes if empty
            print("\nStaging directory removed (clean).")
    except OSError:
        print("\nStaging directory not empty, left for inspection.")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    process_zips()