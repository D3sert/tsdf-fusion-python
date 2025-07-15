
"""
Fuse Carla RGB-D images into a TSDF voxel volume.
"""

import time
import os
import cv2
import numpy as np
import fusion



  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
ddir = 'carla_data/'


def calculate_carla_intrinsics(image_width, image_height, fov_degrees):
    """
    Calculate camera intrinsics matrix from Carla camera parameters.
    
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
    
    return intrinsics


def carla_depth_to_meters(depth_img, max_range_m=60.0):
    """
    Convert Carla depth image to depth in meters.
    Carla encodes depth as RGB where depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    Note: OpenCV loads images as BGR, so we need to handle channel ordering correctly.
    """
    # Ensure we're working with the right data type
    bgr = np.asarray(depth_img, dtype=np.float32)
    
    # Extract BGR channels (OpenCV format) and map to RGB for Carla formula
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    
    # Apply Carla's depth encoding formula: (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    # Note: Using the BGR channels in the RGB formula order
    normalized = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
    
    # Convert to meters
    depth_m = normalized * 1000.0
    
    # Clamp to max range and set invalid depths to 0
    depth_m[depth_m > max_range_m] = 0.0
    
    return depth_m


def fix_carla_camera_pose(carla_pose):
    """
    Convert Carla's world-to-camera matrix to camera-to-world matrix
    and fix coordinate system orientation.
    
    Carla provides world-to-camera transformation matrices, but TSDF fusion
    expects camera-to-world matrices. Additionally, Carla's coordinate system
    needs to be aligned with TSDF's expected orientation.
    """
    # First invert the matrix to convert world-to-camera to camera-to-world
    inverted_pose = np.linalg.inv(carla_pose)
    
    # Apply coordinate system transformation to fix Y-axis curvature
    # Try Y negation to fix downward curvature during left turn
    coord_transform = np.array([
        [ 1,  0,  0,  0],  
        [ 0,  1,  0,  0],  
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float32)
    
    corrected_pose = inverted_pose @ coord_transform
    
    return corrected_pose


if __name__ == "__main__":
    print("Estimating voxel volume bounds...")
    n_imgs = 50  # Use more frames for good reconstruction
    
    # Calculate camera intrinsics from Carla configuration
    # Your Carla config: 'image_size_x': '640', 'image_size_y': '480', 'fov': '90'
    cam_intr = calculate_carla_intrinsics(640, 480, 90.0)
    print("Calculated camera intrinsics:")
    print(cam_intr)
    
    vol_bnds = np.zeros((3,2))
    
    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_path = os.path.join(ddir, "depth_step_%04d.png" % i)
        pose_path = os.path.join(ddir, "matrix_step_%04d.txt" % i)
        
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
        
        # Debug: Print bounds for first few frames
        if i < 3:
            print(f"Frame {i}: bounds min={np.amin(view_frust_pts, axis=1)}, max={np.amax(view_frust_pts, axis=1)}")
            print(f"Current vol_bnds: {vol_bnds}")
    # ======================================================================================================== #
    
    # Validate and clamp volume bounds to reasonable values
    print(f"Raw volume bounds: {vol_bnds}")
    
    # For Carla scenes, center the volume around the first camera position
    first_pose = np.loadtxt(os.path.join(ddir, "matrix_step_0000.txt"))
    camera_pos = first_pose[:3, 3]
    print(f"First camera position: {camera_pos}")
    
    # Use a reasonable volume size (30m x 30m x 15m) centered on the first camera
    half_size_x, half_size_y, half_size_z = 100.0, 100.0, 10.0
    vol_bnds = np.array([
        [camera_pos[0] - half_size_x, camera_pos[0] + half_size_x],
        [camera_pos[1] - half_size_y, camera_pos[1] + half_size_y], 
        [camera_pos[2] - 2.0, camera_pos[2] + half_size_z*2 - 2.0]  # Adjust Z to capture scene above ground
    ], dtype=np.float32)
    
    print(f"Centered volume bounds: {vol_bnds}")
    
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
        pose_path = os.path.join(ddir, "matrix_step_%04d.txt" % i)
        
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