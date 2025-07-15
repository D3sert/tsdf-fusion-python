# TSDF Fusion Issues - Root Cause Analysis

## Critical Issues Causing Disjointed/Repeating Mesh

### 1. INCORRECT CAMERA INTRINSICS (CRITICAL)
- **Problem**: Focal length calculation is incorrect
  - **7-scenes working dataset**: `fx=fy=585` (focal length)
  - **Carla calculation**: `fx=fy=320` (calculated from 90° FOV)
  - **Impact**: ~45% error in focal length completely distorts depth-to-3D projection
- **Location**: `carla_demo.py:115` - `calculate_carla_intrinsics(640, 480, 90.0)`
- **Fix**: Verify Carla's actual camera intrinsics or adjust FOV calculation

### 2. VOLUME BOUNDS COMPLETELY IGNORED (CRITICAL)
- **Problem**: Estimated volume bounds from view frustums are discarded
  - **Working demo**: Uses estimated bounds from view frustums (lines 18-33)
  - **Carla demo**: **Overwrites** estimated bounds with arbitrary fixed volume (lines 164-170)
  - **Impact**: TSDF volume may not contain the actual scene geometry
- **Location**: `carla_demo.py:156-170`
- **Fix**: Use the computed `vol_bnds` from view frustum estimation instead of overriding
- **Note**: The view frustum calculation works correctly, but then gets **discarded**

### 3. COORDINATE SYSTEM MISMATCH (SUSPECTED)
- **Problem**: Coordinate transformation may not be properly converting between systems
  - Fusion expects `cam_pose` as **camera-to-world** matrix
  - Carla provides vehicle poses, then applies `fix_carla_camera_pose()` which:
    - Inverts the matrix (correct)
    - Applies identity transform (does nothing)
- **Location**: `carla_demo.py:97-102`
- **Impact**: Final positioning errors in 3D space
- **Fix**: Determine proper coordinate system transformation between Carla and TSDF fusion

## Why This Causes Disjointed/Repeating Buildings

The **disjointed and repeating** pattern happens because:
1. **Wrong intrinsics** cause incorrect 3D point projection from depth images
2. **Wrong volume bounds** mean geometry gets placed in wrong locations in the TSDF volume
3. **Coordinate issues** cause final positioning errors

**The camera poses themselves might be correct** - the real issues are the preprocessing steps that transform the data before it reaches the fusion algorithm.

## Priority Order for Fixes
1. Fix camera intrinsics calculation
2. Use computed volume bounds instead of overriding them
3. Investigate and fix coordinate system transformation