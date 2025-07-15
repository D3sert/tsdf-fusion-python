# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python implementation of Volumetric TSDF (Truncated Signed Distance Function) Fusion for RGB-D images. The project fuses multiple registered color and depth images into a projective TSDF volume to create high-quality 3D surface meshes and point clouds.

## Development Environment

- **Python Environment**: Use the TSDF Conda environment at `/home/d3s/miniconda3/envs/tsdf/bin/python3.11`
- **GPU Support**: Optional CUDA acceleration via PyCUDA (falls back to CPU mode if unavailable)

## Key Dependencies

- NumPy, OpenCV, Scikit-image, Numba (core processing)
- PyCUDA (optional GPU acceleration)
- Matplotlib (visualization)

## Core Architecture

### Main Components

1. **fusion.py** - Core TSDF fusion implementation
   - `TSDFVolume` class: Main volumetric fusion engine
   - GPU/CPU dual-mode implementation with CUDA kernels
   - Key methods:
     - `integrate()`: Fuses RGB-D frames into TSDF volume
     - `get_mesh()`: Extracts 3D mesh using marching cubes
     - `get_point_cloud()`: Generates point cloud from volume
   
2. **demo.py** - Standard demo using 7-scenes dataset
   - Processes 1000 RGB-D images with 2cm voxel resolution
   - Expected data format: frame-XXXXXX.{color.jpg, depth.png, pose.txt}
   
3. **carla_demo.py** - Carla simulator-specific demo
   - Includes `carla_to_tsdf()` function for Carla depth conversion
   - Processes Carla-generated RGB-D sequences

### Data Structure Expectations

- **RGB Images**: 24-bit PNG/JPG format
- **Depth Images**: 16-bit PNG in millimeters
- **Camera Poses**: 4x4 rigid transformation matrices in .txt files
- **Camera Intrinsics**: 3x3 matrix in camera-intrinsics.txt

## Common Commands

### Running Demos
```bash
# Standard demo (requires data/ directory with 7-scenes format)
python demo.py

# Carla-specific demo (requires carla_data/ directory)
python carla_demo.py
```

### Testing GPU Mode
The code automatically detects CUDA availability and falls back to CPU mode if PyCUDA import fails.

## File Organization

- `data/` - Standard RGB-D dataset (7-scenes format)
- `carla_data/` - Carla simulator RGB-D sequences
- `mesh.ply`, `pc.ply` - Generated 3D outputs
- `dev.ipynb` - Development notebook

## Architecture Notes

- **Dual Mode Processing**: Automatically switches between GPU (CUDA) and CPU (Numba) implementations
- **Memory Management**: Handles large voxel volumes with configurable GPU memory limits
- **Coordinate Systems**: Uses world coordinates with configurable volume bounds
- **Integration Pipeline**: RGB-D frames → camera frustum estimation → TSDF integration → mesh extraction

## Performance Characteristics

- GPU mode: ~30 FPS for RGB-D integration
- CPU mode: ~0.4 FPS for RGB-D integration
- Typical output: High-resolution 3D meshes suitable for visualization in Meshlab