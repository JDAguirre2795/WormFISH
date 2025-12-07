import os
import glob
import numpy as np
import tifffile

# Import big-fish components
try:
    from bigfish import stack
    from bigfish.detection import detect_spots, decompose_dense
    BIGFISH_AVAILABLE = True
except ImportError:
    BIGFISH_AVAILABLE = False
    print("Warning: big-fish not installed. Install with: pip install big-fish")


def process_tiff_file(input_path, output_path, 
                     voxel_size, spot_radius,
                     a647_threshold, a647_alpha, a647_beta, a647_gamma,
                     cy3b_threshold, cy3b_alpha, cy3b_beta, cy3b_gamma):
    """
    Process a single TIFF file for FISH spot detection.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Directory to save spot detection results
        voxel_size: Voxel size in nanometers (z, y, x)
        spot_radius: Spot radius in nanometers (z, y, x)
        a647_threshold: Threshold for A647 channel spot detection
        a647_alpha: Alpha parameter for A647 decomposition
        a647_beta: Beta parameter for A647 decomposition
        a647_gamma: Gamma parameter for A647 decomposition
        cy3b_threshold: Threshold for CY3B channel spot detection
        cy3b_alpha: Alpha parameter for CY3B decomposition
        cy3b_beta: Beta parameter for CY3B decomposition
        cy3b_gamma: Gamma parameter for CY3B decomposition
    """
    if not BIGFISH_AVAILABLE:
        raise ImportError("big-fish package is required. Install with: pip install big-fish")
    
    # Load the image
    img = tifffile.imread(input_path)
    
    # Extract channels
    DAPI = img[:,:,:,2]
    A647 = img[:,:,:,0]
    CY3B = img[:,:,:,1]
    
    # Detect spots and perform decomposition for A647 channel
    print(f"  Processing A647 channel...")
    
    spotsA647, threshold = detect_spots(
        images=A647,
        threshold=a647_threshold,
        return_threshold=True, 
        voxel_size=voxel_size,
        spot_radius=spot_radius)
    
    spots_A647_post, dense_regions, reference_spot = decompose_dense(
        image=A647, 
        spots=spotsA647, 
        voxel_size=voxel_size, 
        spot_radius=spot_radius, 
        alpha=a647_alpha,
        beta=a647_beta,
        gamma=a647_gamma)
    
    print(f"    A647 - before decomposition: {spotsA647.shape}")
    print(f"    A647 - after decomposition: {spots_A647_post.shape}")
    
    # Detect spots and perform decomposition for CY3B channel
    print(f"  Processing CY3B channel...")
    
    spotsCY3B, threshold = detect_spots(
        images=CY3B,
        threshold=cy3b_threshold,
        return_threshold=True, 
        voxel_size=voxel_size,
        spot_radius=spot_radius)
    
    spots_CY3B_post, dense_regions, reference_spot = decompose_dense(
        image=CY3B, 
        spots=spotsCY3B, 
        voxel_size=voxel_size, 
        spot_radius=spot_radius, 
        alpha=cy3b_alpha,
        beta=cy3b_beta,
        gamma=cy3b_gamma)
    
    print(f"    CY3B - before decomposition: {spotsCY3B.shape}")
    print(f"    CY3B - after decomposition: {spots_CY3B_post.shape}")
    
    # Extract the base name of the file
    base_name = os.path.basename(input_path).replace(".tif", "")
    
    # Get the pos folder name (e.g., pos0)
    pos_folder = os.path.basename(os.path.dirname(input_path))
    
    # Create subdirectory in the output path
    subdir = os.path.join(output_path, pos_folder)
    os.makedirs(subdir, exist_ok=True)
    
    # Save the results
    np.save(os.path.join(subdir, f"spots_A647_{base_name}.npy"), spots_A647_post)
    np.save(os.path.join(subdir, f"spots_CY3B_{base_name}.npy"), spots_CY3B_post)
    print(f"  ✓ Saved spots for: {base_name}")


def process_fish_spots(input_root, output_root=None,
                      voxel_size=(1000, 103, 103),
                      spot_radius=(1050, 300, 300),
                      a647_threshold=15, a647_alpha=0.5, a647_beta=1.7, a647_gamma=15,
                      cy3b_threshold=25, cy3b_alpha=0.5, cy3b_beta=1.7, cy3b_gamma=25):
    """
    Process FISH spot detection for ALL files in pos* folders.
    Creates DetectedDots folder if output_root not specified.
    
    Args:
        input_root: Root directory containing TIFF files
        output_root: Root directory to save results (if None, creates DetectedDots)
        voxel_size: Voxel size in nanometers (z, y, x)
        spot_radius: Spot radius in nanometers (z, y, x)
        a647_threshold: Threshold for A647 channel spot detection
        a647_alpha: Alpha parameter for A647 decomposition
        a647_beta: Beta parameter for A647 decomposition
        a647_gamma: Gamma parameter for A647 decomposition
        cy3b_threshold: Threshold for CY3B channel spot detection
        cy3b_alpha: Alpha parameter for CY3B decomposition
        cy3b_beta: Beta parameter for CY3B decomposition
        cy3b_gamma: Gamma parameter for CY3B decomposition
    """
    print("=" * 60)
    print("FISH Spot Detection Pipeline")
    print("=" * 60)
    
    print(f"Voxel size: {voxel_size}")
    print(f"Spot radius: {spot_radius}")
    print(f"A647 Parameters: threshold={a647_threshold}, alpha={a647_alpha}, beta={a647_beta}, gamma={a647_gamma}")
    print(f"CY3B Parameters: threshold={cy3b_threshold}, alpha={cy3b_alpha}, beta={cy3b_beta}, gamma={cy3b_gamma}")
    
    if not BIGFISH_AVAILABLE:
        print("ERROR: big-fish package is not installed.")
        print("Install with: pip install big-fish")
        return
    
    # Set default output directory if not provided
    if output_root is None:
        output_root = os.path.join(input_root, "DetectedDots")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    print(f"\nInput: {input_root}")
    print(f"Output: {output_root}")
    
    # Find all pos* folders directly under input_root
    pos_folders = []
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path) and item.startswith("pos"):
            pos_folders.append(item)
    
    print(f"\nFound {len(pos_folders)} pos folders")
    
    # Process each pos folder
    for pos_folder in sorted(pos_folders):
        pos_path = os.path.join(input_root, pos_folder)
        
        # Look for ALL .tif files in this pos folder
        tiff_files = glob.glob(os.path.join(pos_path, "*.tif"))
        
        print(f"\nProcessing {pos_folder}: {len(tiff_files)} .tif files")
        
        # Process each file in this pos folder
        for i, tiff_file in enumerate(sorted(tiff_files), 1):
            filename = os.path.basename(tiff_file)
            print(f"\n[{i}/{len(tiff_files)}] Processing: {filename}")
            process_tiff_file(
                tiff_file, 
                output_root,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                a647_threshold=a647_threshold,
                a647_alpha=a647_alpha,
                a647_beta=a647_beta,
                a647_gamma=a647_gamma,
                cy3b_threshold=cy3b_threshold,
                cy3b_alpha=cy3b_alpha,
                cy3b_beta=cy3b_beta,
                cy3b_gamma=cy3b_gamma
            )
    
    print(f"\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"Results saved in: {output_root}")
    print("=" * 60)