
"""
Fuse Carla RGB-D images into a TSDF voxel volume.
"""

import time
import os
import cv2
import numpy as np
import fusion
import sys
from omegaconf import OmegaConf



  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #

def get_output_directory_from_config(config_path="../cfg/carla_exp.yaml"):
    """Get the output directory from the config file."""
    try:
        cfg = OmegaConf.load(config_path)
        if hasattr(cfg, 'experiment_results') and hasattr(cfg.experiment_results, 'output_directory'):
            return cfg.experiment_results.output_directory
        else:
            print(f"Warning: No output directory found in config, using default")
            return "../results/latest"
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return "../results/latest"

# Get output directory from config
ddir = get_output_directory_from_config()


def calculate_carla_intrinsics(image_width, image_height, fov_degrees):
    """
    Calculate camera intrinsics matrix from Carla camera parameters.
    
    Based on Carla 9.15 documentation and working 7-scenes dataset comparison:
    - 7-scenes: fx=fy=585 for 640x480 images
    - This calculation should yield similar values for proper TSDF fusion
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels  
        fov_degrees: Horizontal field of view in degrees
    
    Returns:
        3x3 camera intrinsics matrix
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov_degrees)
    
    # Calculate focal length in pixels
    # For horizontal FOV: focal_length = (image_width / 2) / tan(fov/2)
    focal_length = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    
    # Principal point (assume center of image)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Camera intrinsics matrix
    # [fx  0  cx]
    # [ 0 fy  cy] 
    # [ 0  0   1]
    intrinsics = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    print(f"Calculated focal length: {focal_length:.1f} (should be similar to 7-scenes: ~585)")
    return intrinsics


def carla_depth_to_meters(depth_img, max_range_m=60.0):
    """
    Convert Carla depth image to depth in meters.
    
    Based on Carla 9.15 official documentation:
    depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    
    Note: OpenCV loads images as BGR, so we need to handle channel ordering correctly.
    """
    # Ensure we're working with the right data type
    bgr = np.asarray(depth_img, dtype=np.float32)
    
    # Extract BGR channels (OpenCV format) and map to RGB for Carla formula
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    
    # Apply Carla's official depth encoding formula
    # (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    normalized = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
    
    # Convert to meters (Carla formula already includes *1000 scaling)
    depth_m = normalized * 1000.0
    
    # Clamp to max range and set invalid depths to 0
    depth_m[depth_m > max_range_m] = 0.0
    
    return depth_m


def fix_carla_camera_pose(carla_pose):
    """
    Convert Carla's vehicle transformation matrix to camera-to-world matrix
    with proper coordinate system alignment.
    
    Carla vehicle transforms are camera-to-world but need coordinate system fixes:
    - Carla uses UE4 coordinate system: X-forward, Y-right, Z-up
    - TSDF fusion expects standard computer vision: X-right, Y-down, Z-forward
    
    The matrix from generate_carla_files.py is already camera-to-world,
    but needs coordinate system transformation.
    """
    # Apply coordinate transformation from Carla (UE4) to TSDF (OpenCV)
    # Carla: X-forward, Y-right, Z-up
    # OpenCV: X-right, Y-down, Z-forward
    # Transformation: [X_cv, Y_cv, Z_cv] = [Y_carla, -Z_carla, X_carla]
    coord_transform = np.array([
        [ 0,  0,  1,  0],  # X_cv = Z_carla (forward)
        [ 1,  0,  0,  0],  # Y_cv = X_carla (right) 
        [ 0, -1,  0,  0],  # Z_cv = -Y_carla (down)
        [ 0,  0,  0,  1]
    ], dtype=np.float32)
    
    # Apply coordinate transformation
    corrected_pose = carla_pose @ coord_transform
    
    return corrected_pose


if __name__ == "__main__":
    print(f"Using output directory: {ddir}")
    print("Estimating voxel volume bounds...")
    n_imgs = 40  # Use more frames for good reconstruction
    skip = 20
    
    # Calculate camera intrinsics from Carla configuration
    # From log.log: 'width': 640, 'height': 480, 'fov': 90
    cam_intr = calculate_carla_intrinsics(640, 480, 90.0)
    print("Calculated camera intrinsics:")
    print(cam_intr)
    print(f"For comparison, 7-scenes uses fx=fy=585 for 640x480 images")
    
    vol_bnds = np.zeros((3,2))
    
    for i in range(skip,skip+n_imgs):
        # Read depth image and camera pose
        depth_path = os.path.join(ddir, "depth_step_%04d.png" % i)
        pose_path = os.path.join(ddir, "transform_step_%04d.txt" % i)
        
        # Check if files exist
        if not os.path.exists(depth_path):
            print(f"Warning: Depth image {depth_path} not found, skipping...")
            continue
        if not os.path.exists(pose_path):
            print(f"Warning: Pose file {pose_path} not found, skipping...")
            continue
            
        depth_img = cv2.imread(depth_path)
        if depth_img is None:
            print(f"Warning: Could not load depth image {depth_path}, skipping...")
            continue
            
        depth_im = carla_depth_to_meters(depth_img)
        cam_pose = np.loadtxt(pose_path)  # 4x4 rigid transformation matrix
        
        # Fix Carla camera coordinate system (cameras face backward by default)
        cam_pose = fix_carla_camera_pose(cam_pose)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
        
    # ======================================================================================================== #
    
    # Use the computed volume bounds from view frustums (DON'T override them!)
    print(f"Final volume bounds from view frustum estimation: {vol_bnds}")
    
    # Validate bounds are reasonable
    vol_size = vol_bnds[:,1] - vol_bnds[:,0]
    if np.any(vol_size <= 0) or np.any(vol_size > 500):  # Sanity check
        print("WARNING: Volume bounds seem unreasonable, using fallback")
        first_pose = np.loadtxt(os.path.join(ddir, "transform_step_0000.txt"))
        first_pose = fix_carla_camera_pose(first_pose)  # Apply coordinate fix
        camera_pos = first_pose[:3, 3]
        print(f"First camera position (after coord fix): {camera_pos}")
        
        # Reasonable fallback volume (50m x 50m x 20m)
        half_size_x, half_size_y, half_size_z = 25.0, 25.0, 10.0
        vol_bnds = np.array([
            [camera_pos[0] - half_size_x, camera_pos[0] + half_size_x],
            [camera_pos[1] - half_size_y, camera_pos[1] + half_size_y], 
            [camera_pos[2] - 2.0, camera_pos[2] + half_size_z*2 - 2.0]
        ], dtype=np.float32)
        print(f"Using fallback volume bounds: {vol_bnds}")
    else:
        print(f"Using computed volume bounds: {vol_bnds}")
    
    # Calculate volume size for verification
    vol_size = vol_bnds[:,1] - vol_bnds[:,0]
    voxel_size = 0.2  # Use 10cm voxel size for good detail
    vol_dim_estimate = vol_size / voxel_size
    total_voxels = np.prod(vol_dim_estimate)
    
    print(f"Volume size: {vol_size}")
    print(f"Estimated voxel dimensions: {vol_dim_estimate}")
    print(f"Estimated total voxels: {total_voxels:,.0f}")

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i+1, n_imgs))

        # Read RGB-D image and camera pose
        color_path = os.path.join(ddir, "rgb_step_%04d.png" % i)
        depth_path = os.path.join(ddir, "depth_step_%04d.png" % i)
        pose_path = os.path.join(ddir, "transform_step_%04d.txt" % i)
        
        # Check if all required files exist
        if not all(os.path.exists(p) for p in [color_path, depth_path, pose_path]):
            print(f"Warning: Missing files for frame {i}, skipping...")
            continue
            
        # Load images
        color_image = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path)
        
        if color_image is None or depth_img is None:
            print(f"Warning: Could not load images for frame {i}, skipping...")
            continue
            
        # Convert BGR to RGB for color image (OpenCV loads as BGR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Convert Carla depth to meters
        depth_im = carla_depth_to_meters(depth_img)
        
        # Load camera pose and fix coordinate system
        cam_pose = np.loadtxt(pose_path)  # 4x4 rigid transformation matrix
        cam_pose = fix_carla_camera_pose(cam_pose)  # Fix backward-facing camera

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    
    # Debug: Check TSDF volume statistics
    tsdf_vol_data, color_vol_data = tsdf_vol.get_volume()
    print(f"TSDF volume stats: min={tsdf_vol_data.min():.4f}, max={tsdf_vol_data.max():.4f}")
    print(f"TSDF volume valid voxels: {np.sum(tsdf_vol_data != 1.0)}")
    
    try:
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        fusion.meshwrite("mesh.ply", verts, faces, norms, colors)
        print(f"Mesh generated with {len(verts)} vertices and {len(faces)} faces")
    except ValueError as e:
        print(f"Mesh generation failed: {e}")
        print("This usually means no valid surfaces were found in the TSDF volume.")
        print("Possible issues:")
        print("1. Camera poses might be incorrect")
        print("2. Depth images might not be properly converted")
        print("3. Volume bounds might be too large or misaligned")

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    try:
        point_cloud = tsdf_vol.get_point_cloud()
        fusion.pcwrite("pc.ply", point_cloud)
        print(f"Point cloud generated with {len(point_cloud)} points")
    except ValueError as e:
        print(f"Point cloud generation failed: {e}")
        print("Same issue as mesh generation - no valid surfaces found.")