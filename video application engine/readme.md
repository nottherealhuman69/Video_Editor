# Video Chunk Filter & Transition Processor

This module finalizes the video editing pipeline by applying **frame-level filters** and **smooth transitions** between pre-processed video chunks, based on a structured JSON modification file.

## Role in the Pipeline

This script acts as the **final stage** in a multi-part AI video editing system. It takes:
- Segmented video chunks from previous stages
- A `.json` file detailing where and why to apply filters/transitions
- External `.py` or `.ipynb` files defining the transformation logic

And outputs a fully composited, visually enhanced final video (`final_video.mp4`).

---

## What This Code Does

### 1. **Dynamic Loading of Filter/Transition Logic**
- Supports both `.py` and `.ipynb` function definitions
- Auto-loads filters from `filters.py` or `filters.ipynb`
- Auto-loads transitions from `transitions.py` or `transitions.ipynb`

### 2. **Chunk-Wise Processing**
- Iterates through all video chunks (e.g., `chunk_0.mp4`, `chunk_1.mp4`, ...)
- Applies specified filters (grayscale, blur, cartoon, edge_detection, etc.) to selected chunks
- Applies transition effects (crossfade, glitch, fade, zoom, 3D flips, etc.) between defined pairs

### 3. **Reason-Aware Modifications**
- Each filter or transition includes a human-readable rationale in the JSON
- Allows explainable editing decisions for storytelling or design choices

### 4. **Final Video Assembly**
- Combines modified chunks and transitions using frame-overlap blending
- Writes the result into a cohesive, high-quality video file

---

## Expected Directory Structure

```
project-root/
├── chunks/                     # Folder containing chunked videos
│   ├── chunk_0.mp4
│   ├── chunk_1.mp4
│   └── ...
├── filters.py / filters.ipynb  # Filter function definitions
├── transitions.py / transitions.ipynb  # Transition function definitions
├── modifications.json          # JSON file defining filters/transitions
├── run_video_processing.py     # Entry point script
├── video_processor.py          # Core logic
└── final_video.mp4             # Output video
```

---

## Prerequisites

Install required dependencies:

```bash
pip install opencv-python numpy nbformat nbconvert
```

---

## Available Effects

**Filters:** grayscale, sepia, cartoon_effect, pencil_sketch, edge_detection, gaussian_blur, motion_blur, invert_colors, color_tint, emboss_filter, sharpen, bilateral_filter

**Transitions:** crossfade, fade_to_black, slide_left, wipe_right, glitch, zoom_blur, rgb_split, pixel_dissolve, flip_3d_perspective, door_open, tv_static, liquid_ink_splash

---

## How to Run

1. Ensure all input chunks are placed in the `chunks/` folder.
2. Define filters/transitions in `filters.py` and `transitions.py`, or their `.ipynb` equivalents.
3. Add your desired modifications in `modifications.json`.

Then run:

```bash
python run_video_processing.py
```

---
