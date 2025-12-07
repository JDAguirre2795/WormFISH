import os
import glob
import numpy as np
from stardist.models import StarDist3D
from csbdeep.utils import normalize
import tifffile


def process_dapi_channel(input_path, output_root, model):
    """
    Process a single image to extract DAPI channel and predict nuclei.
    
    Args:
        input_path: Path to input TIFF file
        output_root: Root directory to save masks
        model: Loaded StarDist3D model
    """
    # Load the image
    img = tifffile.imread(input_path)
    
    # Extract the DAPI channel
    DAPI = img[:,:,:,2]
    print(f"  DAPI shape: {DAPI.shape}")
    
    # Determine the number of channels
    n_channel = 1 if DAPI.ndim == 3 else DAPI.shape[-1]
    axis_norm = (0,1,2)  # Normalize channels independently
    
    # Normalize the DAPI image
    img_normalized = normalize(DAPI, 1, 99.8, axis=axis_norm)
    
    # Predict instances using the Stardist model
    labels, details = model.predict_instances(img_normalized, n_tiles=(1, 12, 12))
    
    # Get the pos folder name (e.g., pos0)
    pos_folder = os.path.basename(os.path.dirname(input_path))
    base_name = os.path.basename(input_path).replace(".tif", "")
    
    # Create output directory structure
    output_dir = os.path.join(output_root, pos_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the labels
    output_path = os.path.join(output_dir, f"Masks_{base_name}.npy")
    np.save(output_path, labels)
    print(f"  ✓ Saved to: {pos_folder}/{os.path.basename(output_path)}")
    
    return output_path


def process_directory_DAPI_hyb0(input_root, output_root, model):
    """
    Process ONLY files starting with 'hyb_0' in pos* folders directly under input_root.
    Saves masks in NuclearMasks/pos* folders.
    
    Args:
        input_root: Root directory containing TIFF files
        output_root: Root directory to save mask files
        model: Loaded StarDist3D model
    """
    print(f"\nSearching for hyb_0 files in: {input_root}")
    
    # Find all pos* folders directly under input_root
    pos_folders = []
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path) and item.startswith("pos"):
            pos_folders.append(item)
    
    print(f"Found {len(pos_folders)} pos folders: {sorted(pos_folders)}")
    
    total_files = 0
    for pos_folder in sorted(pos_folders):
        pos_path = os.path.join(input_root, pos_folder)
        
        # Look for hyb_0*.tif files in this pos folder
        pattern = os.path.join(pos_path, "hyb_0*.tif")
        files_in_pos = glob.glob(pattern)
        
        print(f"\n{pos_folder}: Found {len(files_in_pos)} hyb_0 files")
        
        for i, tiff_file in enumerate(sorted(files_in_pos), 1):
            filename = os.path.basename(tiff_file)
            print(f"  [{i}/{len(files_in_pos)}] Processing: {filename}")
            process_dapi_channel(tiff_file, output_root, model)
            total_files += 1
    
    print(f"\nTotal files processed: {total_files}")


def process_directory_all_hyb(input_root, output_root, model):
    """
    Process ALL hyb_* files in pos* folders directly under input_root.
    Saves masks in NuclearMasks/pos* folders.
    
    Args:
        input_root: Root directory containing TIFF files
        output_root: Root directory to save mask files
        model: Loaded StarDist3D model
    """
    print(f"\nSearching for ALL hyb files in: {input_root}")
    
    # Find all pos* folders directly under input_root
    pos_folders = []
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path) and item.startswith("pos"):
            pos_folders.append(item)
    
    print(f"Found {len(pos_folders)} pos folders: {sorted(pos_folders)}")
    
    total_files = 0
    for pos_folder in sorted(pos_folders):
        pos_path = os.path.join(input_root, pos_folder)
        
        # Look for ALL hyb_*.tif files in this pos folder
        pattern = os.path.join(pos_path, "hyb_*.tif")
        files_in_pos = glob.glob(pattern)
        
        print(f"\n{pos_folder}: Found {len(files_in_pos)} hyb files")
        
        for i, tiff_file in enumerate(sorted(files_in_pos), 1):
            filename = os.path.basename(tiff_file)
            print(f"  [{i}/{len(files_in_pos)}] Processing: {filename}")
            process_dapi_channel(tiff_file, output_root, model)
            total_files += 1
    
    print(f"\nTotal files processed: {total_files}")


def process_all_images(input_root, output_root=None, model_path=None, model_name='stardistV2.6', 
                       process_all_hyb=False):
    """
    Complete pipeline to process all images in pos* folders.
    
    Args:
        input_root: Root directory containing TIFF files
        output_root: Root directory to save mask files (if None, creates NuclearMasks)
        model_path: Path to model directory (if None, uses default)
        model_name: Name of the StarDist model
        process_all_hyb: If True, process ALL hyb files. If False, process only hyb_0
    """
    print("=" * 60)
    print("StarDist 3D Nuclear Segmentation")
    print("=" * 60)
    
    # Set default output directory if not provided
    if output_root is None:
        output_root = os.path.join(input_root, "NuclearMasks")
        print(f"Output directory not specified, using: {output_root}")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    print(f"Saving masks to: {output_root}")
    
    # Load model
    print(f"\nLoading model {model_name}...")
    model = StarDist3D(None, name=model_name, basedir=model_path)
    
    # Process images
    print(f"\nProcessing images from: {input_root}")
    
    if process_all_hyb:
        print("Mode: Processing ALL hyb_* files")
        process_directory_all_hyb(input_root, output_root, model)
    else:
        print("Mode: Processing ONLY hyb_0 files")
        process_directory_DAPI_hyb0(input_root, output_root, model)
    
    print(f"\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"Masks saved in: {output_root}")
    print("=" * 60)


def check_folder_structure(input_root):
    """
    Check the folder structure and available files.
    
    Args:
        input_root: Root directory to check
    """
    print(f"\nChecking folder structure: {input_root}")
    
    if not os.path.exists(input_root):
        print(f"ERROR: Path does not exist: {input_root}")
        return
    
    # Find all pos* folders
    pos_folders = []
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path) and item.startswith("pos"):
            pos_folders.append(item)
    
    print(f"Found {len(pos_folders)} pos folders")
    
    for pos_folder in sorted(pos_folders):
        pos_path = os.path.join(input_root, pos_folder)
        
        # Count files
        hyb_0_files = glob.glob(os.path.join(pos_path, "hyb_0*.tif"))
        all_hyb_files = glob.glob(os.path.join(pos_path, "hyb_*.tif"))
        
        print(f"\n{pos_folder}:")
        print(f"  hyb_0*.tif files: {len(hyb_0_files)}")
        print(f"  All hyb_*.tif files: {len(all_hyb_files)}")
        
        if all_hyb_files:
            print(f"  Files:")
            for f in sorted(all_hyb_files):
                filename = os.path.basename(f)
                size_mb = os.path.getsize(f) / (1024 * 1024)
                print(f"    - {filename} ({size_mb:.1f} MB)")