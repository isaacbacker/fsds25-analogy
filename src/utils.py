#!/usr/bin/env python3
"""
Utility functions for downloading and extracting files.

These utilities are preserved for manual model downloading if needed,
though the main workflow now uses gensim's downloader API.
"""

import os
import gzip
import zipfile
import requests
from tqdm import tqdm


def download_file(url, destination):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        # Stream the download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)
        
        print(f"✓ Download complete: {destination}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extraction complete")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting zip file: {e}")
        return False


def extract_gzip(gz_path, output_path):
    """
    Extract a gzip file using chunked reading for memory efficiency.
    
    Args:
        gz_path: Path to the gzip file
        output_path: Path for the extracted file
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    import shutil
    
    try:
        print(f"Extracting {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extraction complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting gzip file: {e}")
        return False


def ensure_directory(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        str: Absolute path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    return os.path.abspath(directory)


def get_file_size_mb(filepath):
    """
    Get the size of a file in megabytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        float: File size in MB, or None if file doesn't exist
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return None


def file_exists(filepath):
    """
    Check if a file exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.exists(filepath) and os.path.isfile(filepath)
