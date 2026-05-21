
## Environment Setup

Three conda/pip environment files are provided depending on your use case:

| Environment | GPU | Use Case |

| `environment_analysis_GPU.yml` |  Yes | Data analysis with GPU acceleration |
| `environment_analysis.yml` | No | Data analysis on CPU only |
| `enironment_annotation.yml` | — | Cell nuclei annotation via Napari |

Install your chosen environment with:
```bash
conda env create -f <environment_file.yml>
conda activate <env_name>
```
---

##  Notebooks

### 1. Nuclear Mask Annotation & 3D Model Training

Run notebooks in the following order:

**Step 1 — Annotate training images**
```
Napari_NuclearMask_Annotation.ipynb
```
Use Napari to annotate nuclear masks. All training images must be annotated before proceeding.

**Step 2 — Train the 3D model**
```
StarDist_3D_Model_training.ipynb
```
Run after all training images have been annotated to train a StarDist 3D segmentation model.

**Step 3 — Run segmentation**
```
StarDist_Nuclear_Segmentation.ipynb
```
Apply the trained model to segment nuclei in your dataset.

---

### 2. Neuron Annotation

**Step 1 — Generate HDF5 files**
```
HDF5_generation_and_coordinates_extraction.ipynb
```
Prepares data in HDF5 format required for annotation.

**Step 2 — Annotate neurons**

Annotation is performed using the **TargetTrack** GUI:
>  [https://github.com/rahi-lab/targettrack](https://github.com/rahi-lab/targettrack)

**Step 3 — Extract neuron coordinates**

Return to `HDF5_generation_and_Coordinates_extraction.ipynb` to extract the 3D coordinates of each annotated neuron point.

---

### 3. Dot Detection

**Detect RNA dots**
```
DotDetection.ipynb
```
Run this notebook to perform dot detection on your imaging data.

---

### 4. Dot Assignment & Visualization

**Assign dots**
```
dot_assignment.ipynb
```
Assigns detected dots to their corresponding cells/nuclei.

**Visualize results**
```
dot_plotting.ipynb
```
Generates plots to visualize the detected and assigned dots.

---


