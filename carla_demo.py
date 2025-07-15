
"""
Fuse Carla RGB-D images into a TSDF voxel volume.
"""

import time
import os
import cv2
import numpy as np
from scipy.spatial.transform import RigidTransform as Tf
import fusion
import matplotlib.pyplot as plt



  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
ddir = 'carla_data/'


def carla_to_tsdf(img, max_range_m=60.0):
    rgb    = np.asarray(img, dtype=np.uint32)
    r, g, b = rgb[..., 2], rgb[..., 1], rgb[..., 0]
    n  = (r + g*256 + b*256*256) / (256**3 - 1)

    depth_m = n * 1000.0 
    depth_m[depth_m > max_range_m] = 0.0
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    return depth_mm


print("Estimating voxel volume bounds...")
n_imgs = 49
cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
vol_bnds = np.zeros((3,2))
for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread(os.path.join(ddir, "depth_step_%04d.png"%(i)))
    depth_im = carla_to_tsdf(depth_im)
    #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt(os.path.join(ddir, "matrix_step_%04d.txt"%(i)))  # 4x4 rigid transformation matrix

# Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
# ======================================================================================================== #

# ======================================================================================================== #
# Integrate
# ======================================================================================================== #
    # Initialize voxel volume
print("Initializing voxel volume...")
tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

# Loop through RGB-D images and fuse them together
t0_elapse = time.time()
for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.imread(os.path.join(ddir, "rgb_step_%04d.png"%(i)))
    depth_im = cv2.imread(os.path.join(ddir, "depth_step_%04d.png"%(i)))
    depth_im = carla_to_tsdf(depth_im)
    cam_pose = np.loadtxt(os.path.join(ddir, "matrix_step_%04d.txt"%(i)))  # 4x4 rigid transformation matrix

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

fps = n_imgs / (time.time() - t0_elapse)
print("Average FPS: {:.2f}".format(fps))

# Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
print("Saving mesh to mesh.ply...")
verts, faces, norms, colors = tsdf_vol.get_mesh()
fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

# Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
print("Saving point cloud to pc.ply...")
point_cloud = tsdf_vol.get_point_cloud()
fusion.pcwrite("pc.ply", point_cloud)