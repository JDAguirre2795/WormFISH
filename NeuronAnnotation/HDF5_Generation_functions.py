
"""
Utilities for converting TIFF image stacks into HDF5 datasets.

These functions are part of the WormSeqFISH pipeline and are designed to
convert raw imaging data into a standardized H5 format for downstream
processing.
"""

import os
import re
import numpy as np
import tifffile
import h5py


def tiff_to_h5(tiff_filepath, h5_filepath, dataset_name="data"):
    """
    Convert a TIFF stack into an HDF5 dataset.

    Parameters
    ----------
    tiff_filepath : str
        Path to the input TIFF file.
    h5_filepath : str
        Path where the output H5 file will be saved.
    dataset_name : str, optional
        Name of the dataset inside the H5 file.

    Returns
    -------
    str
        Path to the written H5 file.
    """
    # Load TIFF image
    tiff_data = tifffile.imread(tiff_filepath)

    # Reshape TIFF data if needed (assumes Z,C,Y,X → C,Y,X,Z)
    # Adjust this reshape depending on your TIFF conventions.
    try:
        reshaped = np.transpose(tiff_data, (1, 2, 3, 0))
    except Exception:
        raise ValueError(
            f"Failed to reshape TIFF data from file: {tiff_filepath}. "
            "Check expected TIFF dimensions."
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(h5_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Save to HDF5
    with h5py.File(h5_filepath, "w") as f:
        f.create_dataset(dataset_name, data=reshaped)

    return h5_filepath


def process_folders(root_folder):
    """
    Scan a root folder for TIFF files and convert them into H5 files.

    Expects TIFF filenames with pattern: `hyb_###_pos##.tif`
    Each pos is saved into its own subfolder under "raw H5 files".

    Parameters
    ----------
    root_folder : str
        Directory containing folders of TIFF files.

    Returns
    -------
    list of str
        List of paths to the created H5 files.
    """
    output_root = os.path.join(root_folder, "raw H5 files")
    os.makedirs(output_root, exist_ok=True)

    created_files = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith((".tif", ".tiff")):
                continue

            # Match hyb_(number)_pos(number)
            match = re.search(r"hyb_(\d+)_pos(\d+)", file_name)
            if match is None:
                print(f"Skipping {file_name}: filename does not match expected pattern.")
                continue

            hyb_number = int(match.group(1))
            pos_number = match.group(2)

            # Build output path
            pos_folder = os.path.join(output_root, f"pos{pos_number}")
            os.makedirs(pos_folder, exist_ok=True)

            h5_name = f"hyb_{hyb_number:03d}_pos{pos_number}.h5"
            h5_path = os.path.join(pos_folder, h5_name)

            # Convert TIFF → H5
            tiff_path = os.path.join(folder_path, file_name)
            created = tiff_to_h5(tiff_path, h5_path)
            created_files.append(created)

    return created_files


def list_h5_files(directory):
    """
    List all H5 files in the given directory, sorted numerically by hyb number.

    Parameters
    ----------
    directory : str
        Directory containing H5 files.

    Returns
    -------
    list of str
        Full paths to H5 files sorted by hyb number.
    """
    files = [f for f in os.listdir(directory) if f.endswith(".h5")]
    
    # Sort by number after 'hyb_'
    def extract_hyb_number(filename):
        match = re.search(r"hyb_(\d+)_", filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Push unmatched files to the end
    
    files.sort(key=extract_hyb_number)
    
    return [os.path.join(directory, f) for f in files]


def merge_h5_files(directory, output_file, N_points=200, C=3, W=2048, H=2048, D=77, clip_quantiles=(0, 1)):
    """
    Merge all H5 files in a directory into a single H5 file with configurable parameters.

    Parameters
    ----------
    directory : str
        Directory containing H5 files to merge.
    output_file : str
        Path to the output merged H5 file.
    N_points : int, optional
        Number of points per frame (default 200).
    C : int, optional
        Number of channels (default 3).
    W : int, optional
        Width of the images (default 2048).
    H : int, optional
        Height of the images (default 2048).
    D : int, optional
        Depth/Z dimension of images (default 77).
    clip_quantiles : tuple(float, float), optional
        Percentiles for clipping 16-bit images before scaling to 8-bit (default (0,1)).

    Returns
    -------
    None
    """
    file_paths = list_h5_files(directory)
    T = len(file_paths)

    if T == 0:
        print("No H5 files found in the directory.")
        return

    q_low, q_high = clip_quantiles

    with h5py.File(output_file, "a") as out_h5:
        for i, file_path in enumerate(file_paths):
            with h5py.File(file_path, "r") as in_h5:
                print(f"Processing {file_path} ({i+1}/{T})")
                image_16bit_all_channels = np.array(in_h5['data'])
                image_8bit_all_channels = []

                for image_16bit in image_16bit_all_channels:
                    # Clip based on quantiles
                    p_low, p_high = np.quantile(image_16bit, [q_low, q_high])
                    image_clipped = np.clip(image_16bit, p_low, p_high)
                    # Scale to 8-bit
                    image_8bit = ((image_clipped - image_clipped.min()) / 
                                  (image_clipped.max() - image_clipped.min()) * 255).astype(np.uint8)
                    image_8bit_all_channels.append(image_8bit)

                image_8bit_all_channels = np.array(image_8bit_all_channels)
                ds = out_h5.create_dataset(f"{i}/frame",
                                           shape=image_8bit_all_channels.shape,
                                           dtype=np.uint8,
                                           compression="lzf")
                ds[...] = image_8bit_all_channels

        # Create points dataset
        points = np.full((T, N_points + 1, 3), np.nan, dtype=np.float32)
        points *= np.array([W, H, D])[None, None, :]
        ds_points = out_h5.create_dataset("points", shape=points.shape, dtype=points.dtype)
        ds_points[...] = points

        # Set global attributes
        out_h5.attrs.update({
            "N_points": N_points,
            "T": T,
            "C": C,
            "W": W,
            "H": H,
            "D": D,
            "description": "Merged H5 dataset"
        })

    print(f"Compiled H5 file created at: {output_file}")

def merge_all_positions(root_folder, N_points=200, C=3, W=2048, H=2048, D=77, clip_quantiles=(0,1)):
    """
    Merge H5 files in each pos# folder inside 'raw H5 files' and save them into a single 'MergedH5' folder.

    Parameters
    ----------
    root_folder : str
        Path to your main folder containing 'raw H5 files'.
    N_points, C, W, H, D : int
        Dataset parameters for merge_h5_files.
    clip_quantiles : tuple(float, float)
        Percentiles for clipping 16-bit images before scaling to 8-bit.
    """
    import os
    
    raw_folder = os.path.join(root_folder, "raw H5 files")
    merged_folder = os.path.join(root_folder, "MergedH5")
    os.makedirs(merged_folder, exist_ok=True)

    for pos_folder_name in os.listdir(raw_folder):
        pos_folder_path = os.path.join(raw_folder, pos_folder_name)
        if not os.path.isdir(pos_folder_path):
            continue

        merged_output = os.path.join(merged_folder, f"merged{pos_folder_name}.h5")
        print(f"Merging H5 files in {pos_folder_name} → {merged_output}")

        merge_h5_files(
            directory=pos_folder_path,
            output_file=merged_output,
            N_points=N_points,
            C=C,
            W=W,
            H=H,
            D=D,
            clip_quantiles=clip_quantiles
        )

    print("All positions merged successfully!")

def extract_all_coordinates(root_folder):
    """
    Extract XYZ coordinates with frame index from all merged H5 files in 'MergedH5'
    and save them as .npy files in 'AnnotationCoordinates' folder.

    Parameters
    ----------
    root_folder : str
        Path to the folder containing 'MergedH5'.
    """
    merged_folder = os.path.join(root_folder, "MergedH5")
    coords_folder = os.path.join(root_folder, "AnnotationCoordinates")
    os.makedirs(coords_folder, exist_ok=True)

    for merged_file in os.listdir(merged_folder):
        if not merged_file.endswith(".h5"):
            continue

        pos_name = merged_file.replace("merged", "").replace(".h5", "")
        output_npy = os.path.join(coords_folder, f"AnnotationCoordinates{pos_name}.npy")
        h5_path = os.path.join(merged_folder, merged_file)

        # Extract coordinates
        with h5py.File(h5_path, 'r') as f:
            points_dataset = f['points'][:]  # shape (T, N_points+1, 3)
            T = f.attrs['T']
            N_points = f.attrs['N_points']
            coords = points_dataset[:, :N_points, :]
            frame_indices = np.arange(T).reshape(-1, 1, 1)
            frame_indices = np.tile(frame_indices, (1, N_points, 1))
            coords_with_frame = np.concatenate([coords, frame_indices], axis=2)
            coords_with_frame = coords_with_frame.reshape(-1, 4)

        # Save as .npy
        np.save(output_npy, coords_with_frame)
        print(f"Saved {output_npy}")

    print("All coordinates extracted successfully!")
