#!/usr/bin/env python3

import cv2
import numpy as np
import fusion

def carla_depth_to_meters(depth_img, max_range_m=60.0):
    """Convert Carla depth image to depth in meters."""
    rgb = np.asarray(depth_img, dtype=np.float32)
    b, g, r = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    normalized = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
    depth_m = normalized * 1000.0
    depth_m[depth_m > max_range_m] = 0.0
    return depth_m

def calculate_carla_intrinsics(image_width, image_height, fov_degrees):
    """Calculate camera intrinsics matrix from Carla camera parameters."""
    fov_rad = np.radians(fov_degrees)
    focal_length = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    intrinsics = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return intrinsics

if __name__ == "__main__":
    # Load test data
    color_image = cv2.cvtColor(cv2.imread("carla_data/rgb_step_0000.png"), cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread("carla_data/depth_step_0000.png")
    depth_im = carla_depth_to_meters(depth_img)
    cam_pose = np.loadtxt("carla_data/matrix_step_0000.txt")
    cam_intr = calculate_carla_intrinsics(640, 480, 90.0)
    
    print("=== DEBUG INFO ===")
    print(f"Color image shape: {color_image.shape}")
    print(f"Color image range: {color_image.min()} - {color_image.max()}")
    print(f"Depth image shape: {depth_im.shape}")
    print(f"Depth image range: {depth_im.min():.3f} - {depth_im.max():.3f}")
    print(f"Valid depth pixels: {np.sum(depth_im > 0)}")
    print(f"Camera pose translation: {cam_pose[:3, 3]}")
    print(f"Camera intrinsics:\n{cam_intr}")
    
    # Test view frustum calculation
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    print(f"View frustum points shape: {view_frust_pts.shape}")
    print(f"View frustum X range: {view_frust_pts[0].min():.3f} - {view_frust_pts[0].max():.3f}")
    print(f"View frustum Y range: {view_frust_pts[1].min():.3f} - {view_frust_pts[1].max():.3f}")
    print(f"View frustum Z range: {view_frust_pts[2].min():.3f} - {view_frust_pts[2].max():.3f}")
    
    # Create small volume for testing
    center = cam_pose[:3, 3]
    vol_bnds = np.array([
        [center[0] - 5, center[0] + 5],
        [center[1] - 5, center[1] + 5], 
        [center[2] - 2, center[2] + 8]
    ], dtype=np.float32)
    
    print(f"Test volume bounds: {vol_bnds}")
    
    # Initialize small TSDF volume
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.1)
    
    # Try integration
    print("Attempting integration...")
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.0)
    
    # Check results
    tsdf_data, color_data = tsdf_vol.get_volume()
    print(f"TSDF volume stats: min={tsdf_data.min():.4f}, max={tsdf_data.max():.4f}")
    print(f"Valid TSDF voxels: {np.sum(tsdf_data != 1.0)}")
    print(f"TSDF values near 0: {np.sum(np.abs(tsdf_data) < 0.1)}")