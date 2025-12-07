import os
import sys
import numpy as np
import tifffile
import napari
from typing import List, Optional
from pathlib import Path


class InterpolationAnnotator:
    def __init__(self):
        """Initialize annotator without automatically creating viewer"""
        self.viewer = None
        self.current_mask_path = None
        self.image_paths = []
        self.output_dir = ""
        
    def start_annotation(self, image_paths: List[str], output_dir: str) -> None:
        """
        Start interactive annotation session
        
        Args:
            image_paths: List of image file paths to annotate
            output_dir: Directory to save annotated masks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out already processed images
        self.image_paths = [
            p for p in image_paths 
            if not os.path.exists(
                os.path.join(output_dir, f"{Path(p).stem}_interpolated.tif")
            )
        ]
        
        self.output_dir = output_dir
        
        if not self.image_paths:
            print("All images have already been annotated!")
            return
            
        # Create viewer
        self.viewer = napari.Viewer()
        self._setup_key_bindings()
        self._load_next_image()
        
    def _setup_key_bindings(self) -> None:
        """Setup keyboard shortcuts"""
        @self.viewer.bind_key('s')
        def _save_and_continue(_):
            self._save_interpolated_mask()
            self._load_next_image()
                
        @self.viewer.bind_key('q')
        def _quit(_):
            print("Annotation complete!")
            self.viewer.close()
    
    def _load_next_image(self) -> None:
        """Load next unprocessed image into viewer"""
        if not self.image_paths:
            if self.viewer:
                self.viewer.close()
            return
            
        image_path = self.image_paths.pop(0)
        image = tifffile.imread(image_path)
        self.current_mask_path = os.path.join(
            self.output_dir,
            f"{Path(image_path).stem}_interpolated.tif"
        )
        
        self.viewer.layers.clear()
        self.viewer.add_image(image, name='Image')
        self.viewer.add_labels(
            np.zeros(image.shape, dtype=np.uint16),
            name='Labels'
        )
        print(f"\nAnnotating: {os.path.basename(image_path)}")
        print("1. Draw on some slices in 'Labels' layer")
        print("2. Click 'Interpolate' button (top-right) to generate interpolated labels")
        print("3. Press [s] to save and continue | [q] to quit")
    
    def _save_interpolated_mask(self) -> None:
        """Save the interpolated mask to disk"""
        if not self.viewer:
            return
            
        try:
            for layer in self.viewer.layers:
                if layer.name == 'Labels - interpolated':
                    tifffile.imwrite(
                        self.current_mask_path,
                        layer.data.astype(np.uint16)
                    )
                    print(f"✓ Saved: {os.path.basename(self.current_mask_path)}")
                    return
            
            print("No interpolated labels found!")
            print("Please:")
            print("  1. Annotate some slices in 'Labels' layer")
            print("  2. Click the 'Interpolate' button")
        except Exception as e:
            print(f"Error saving mask: {str(e)}")


def annotate_folder(input_dir: str, output_dir: str) -> None:
    """
    Convenience function to annotate all TIFF files in a folder
    
    Args:
        input_dir: Directory containing TIFF images
        output_dir: Directory to save annotated masks
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    tiff_files = sorted([
        str(p) for p in input_path.glob("*") 
        if p.suffix.lower() in ['.tif', '.tiff']
    ])
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to annotate")
    annotator = InterpolationAnnotator()
    annotator.start_annotation(tiff_files, output_dir)
    napari.run()  # Start napari event loop

def check_annotation_status(input_dir: Path, output_dir: Path):
    """Check which images have been annotated."""
    # Find all TIFF files
    input_files = sorted([
        f for f in input_dir.glob("*") 
        if f.suffix.lower() in ['.tif', '.tiff']
    ])
    
    # Find existing annotations
    output_files = list(output_dir.glob("*_interpolated.tif"))
    annotated_stems = {f.stem.replace('_interpolated', '') for f in output_files}
    
    # Print summary
    print(f"Total input files: {len(input_files)}")
    print(f"Already annotated: {len(output_files)}")
    print(f"Pending annotation: {len(input_files) - len(output_files)}")
    
    if input_files:
        print("\nInput files:")
        for f in input_files:
            status = "✓" if f.stem in annotated_stems else "□"
            print(f"  {status} {f.name}")
    
    return input_files, output_files