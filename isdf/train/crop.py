import open3d as o3d
import numpy as np
from datetime import datetime
import cv2
import os
import csv
import tf_transformations
from geometry_msgs.msg import TransformStamped
from scipy.spatial import cKDTree

# mesh = o3d.io.read_triangle_mesh("/home/luke/iSDF/results/iSDF/OSVP/meshes/20.000.ply")
# bbox = mesh.get_axis_aligned_bounding_box()
# min_bound = np.array(bbox.min_bound)
# max_bound = np.array(bbox.max_bound)

# ### OSVP+CART
# # min_bound[0] += 0.3
# # min_bound[1] += 0.35
# # min_bound[2] += 0.29

# # max_bound[0] -= 0.05
# # max_bound[1] -= 0.62
# # max_bound[2] += 0

# ### OSVP

# min_bound[0] += 0.3
# min_bound[1] += 0.35
# min_bound[2] += 0.13

# max_bound[0] -= 0.05
# max_bound[1] -= 0.525
# max_bound[2] += 0

# bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
# cropped = mesh.crop(bbox)

# # mesh.paint_uniform_color([0, 1, 0])
# # cropped.paint_uniform_color([1, 0, 0])
# # o3d.visualization.draw_geometries([mesh, cropped])
# # o3d.visualization.draw_geometries([cropped])
# o3d.io.write_triangle_mesh("/home/luke/results/OSVP/meshes/1.ply", cropped)

def transform_stamped_to_matrix(transform_stamped: TransformStamped):
    """
    Converts a geometry_msgs/TransformStamped to a 4x4 transformation matrix.

    Args:
        transform_stamped (TransformStamped): The TransformStamped message.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Extract translation
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    tz = transform_stamped.transform.translation.z

    # Extract rotation (quaternion)
    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w

    # Create a translation matrix
    translation_matrix = tf_transformations.translation_matrix([tx, ty, tz])

    # Create a rotation matrix from the quaternion
    rotation_matrix = tf_transformations.quaternion_matrix([qx, qy, qz, qw])

    # Combine translation and rotation into a single 4x4 transformation matrix
    transformation_matrix = tf_transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    return transformation_matrix

def extract_number(filename):
    base = os.path.splitext(filename)[0]
    num = ''.join(filter(str.isdigit, base))
    return int(num) if num else float('inf')

def project_points_to_image(points_3d, camera_intrinsics, camera_pose):
    points_in_camera = (camera_pose[:3, :3] @ points_3d.T).T + camera_pose[:3, 3]
    z = points_in_camera[:, 2]

    points_2d = np.dot(camera_intrinsics, points_in_camera.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2].reshape(-1, 1)

    return points_2d, z

def filter_visible_points(points_3d, camera_intrinsics, camera_pose, image_width, image_height):
    points_2d, z = project_points_to_image(points_3d, camera_intrinsics, camera_pose)

    visible_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height) & \
                   (z > 0)
    
    return points_3d[visible_mask]

def crop_mesh(path):

    min_bound = np.array([0.15, -0.20, 0.008])
    max_bound = np.array([0.4, 0.05, 0.25])

    mesh = o3d.io.read_triangle_mesh(path)
    # bbox = mesh.get_axis_aligned_bounding_box()
    # min_bound = np.array(bbox.min_bound)
    # max_bound = np.array(bbox.max_bound)

    # ### OSVP+CART
    # min_bound[0] += 0.3
    # min_bound[1] += 0.35
    # min_bound[2] += 0.29

    # max_bound[0] -= 0.05
    # max_bound[1] -= 0.62
    # max_bound[2] += 0

    # print(min_bound)
    # print(max_bound)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped = mesh.crop(bbox)

    mesh.paint_uniform_color([1, 0, 0])
    cropped.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([mesh, cropped])

def normalise_pcd(pcd):
    points = np.asarray(pcd.points)

    centroid = points.mean(axis=0)
    points -= centroid
    max_dist = np.linalg.norm(points, axis=1).max()
    points /= max_dist

    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def calculate_fscore(pred_points, gt_points, threshold=0.1):

    pred_kd_tree = cKDTree(pred_points)
    gt_kd_tree = cKDTree(gt_points)

    # One-direction distances
    dist_pred_to_gt, _ = pred_kd_tree.query(gt_points)
    dist_gt_to_pred, _ = gt_kd_tree.query(pred_points)

    # Count within threshold
    recall = np.mean(dist_pred_to_gt < threshold)
    precision = np.mean(dist_gt_to_pred < threshold)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall


def evaluate(predicted_folder_path, ground_truth_path):

    # Bounding box values for taurus
    # min_bound = np.array([0.24444446, -0.09556114, 0.01222221])
    # max_bound = np.array([0.36870084, 0.04666669, 0.19146334])
    # min_bound = np.array([0.24444446, -0.09556114, 0])
    # max_bound = np.array([0.36870084, 0.04666669, 0.19146334])

    # Bounding box values for pitcher
    # min_bound = np.array([0.21, -0.15, 0.011])
    # max_bound = np.array([0.379, 0.01, 0.25])

    # Bounding box values for tissue box
    min_bound = np.array([0.15, -0.20, 0.012])
    max_bound = np.array([0.4, 0.05, 0.25])

    camera_transform = TransformStamped()
    camera_transform.transform.translation.x = 0.37865784764289856
    camera_transform.transform.translation.y = 0.02831578440964222
    camera_transform.transform.translation.z = 0.33566945791244507
    camera_transform.transform.rotation.x = 0.7081922888755798
    camera_transform.transform.rotation.y = -0.70561283826828
    camera_transform.transform.rotation.z = 0.016810093075037003
    camera_transform.transform.rotation.w = -0.017076168209314346

    camera_pose = transform_stamped_to_matrix(camera_transform)
    # print(camera_pose)

    camera_intrinsics = np.array(
                [
                    [337.2, 0.0, 320],
                    [0.0, 324.73, 180],
                    [0.0, 0.0, 1.0],
                ]
            )
    
    image_height = 360
    image_width = 640

    # Read in ground-truth mesh
    gt = o3d.io.read_triangle_mesh(ground_truth_path)

    # Sample ground-truth mesh
    sampled_gt_pcd = gt.sample_points_uniformly(number_of_points=10000)

    sampled_gt_pcd = normalise_pcd(sampled_gt_pcd)

    # Convert to numpy array for calculations
    gt_pcd = np.asarray(sampled_gt_pcd.points)

    # Get points visible in image
    visible_gt_pcd = filter_visible_points(gt_pcd, camera_intrinsics, camera_pose, image_width, image_height)
    # print(type(visible_gt_pcd))

    # Convert back to point cloud
    gt_pcd_new = o3d.geometry.PointCloud()
    gt_pcd_new.points = o3d.utility.Vector3dVector(visible_gt_pcd)
    o3d.visualization.draw_geometries([sampled_gt_pcd])

    # Create output folder for cropped meshes
    folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = "/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/Evaluation/cropped_mesh/" + folder_timestamp
    os.mkdir(out_folder)
    mesh_out_folder = out_folder + "/meshes"
    pcl_out_folder = out_folder + "/pcl"
    os.mkdir(mesh_out_folder)
    os.mkdir(pcl_out_folder)
    
    # Loop through all intervals of meshes
    ply_files = [f for f in os.listdir(predicted_folder_path)]
    sorted_ply = sorted(ply_files, key=extract_number)

    if 'OSVP' in predicted_folder_path:
        run_type = 'osvp'
    elif 'extended' in predicted_folder_path:
        run_type = 'extended'

    with open(out_folder + f'/{run_type}_results.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Filename', 'Chamfer Distance', 'Hausdorff Distance', 'F-Score', 'Precision', 'Recall'])

        for filename in sorted_ply:

            # Construct path
            predicted_path = os.path.join(predicted_folder_path, filename)

            # Read in predicted mesh
            predicted = o3d.io.read_triangle_mesh(predicted_path)
            
            # Create bounding box
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

            # Apply bounding box
            cropped = predicted.crop(bbox)
            mesh_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cropped_mesh_filename = f"/cropped_{filename}"
            o3d.io.write_triangle_mesh(mesh_out_folder + cropped_mesh_filename, cropped)
            cropped_pcl_filename = f"/cropped_pcl_{filename}"

            # Sample predicted mesh
            sampled_predicted_pcd = cropped.sample_points_uniformly(number_of_points=10000)
            
            # # Normalise to origin
            # center = sampled_predicted_pcd.mean(axis=0)
            # sampled_predicted_pcd -= center

            # # Scale to unit
            # max_dist = np.linalg.norm(sampled_predicted_pcd, axis=1).max()
            # sampled_predicted_pcd /= max_dist

            sampled_predicted_pcd = normalise_pcd(sampled_predicted_pcd)

            o3d.io.write_point_cloud(pcl_out_folder + cropped_pcl_filename, sampled_predicted_pcd)
            # o3d.visualization.draw_geometries([sampled_predicted_pcd])

            # Convert to numpy array for calculations
            predicted_pcd = np.asarray(sampled_predicted_pcd.points)

            # Get points visible in image
            # visible_predicted_pcd = filter_visible_points(predicted_pcd, camera_intrinsics, camera_pose, image_width, image_height)

            # predicted_pcd_new = o3d.geometry.PointCloud()
            # predicted_pcd_new.points = o3d.utility.Vector3dVector(visible_predicted_pcd)
            # o3d.visualization.draw_geometries([sampled_predicted_pcd])
            # o3d.visualization.draw_geometries([predicted_pcd_new])

            # Chamfer Distance
            # d1 = np.asarray(predicted_pcd_new.compute_point_cloud_distance(gt_pcd_new))
            # d2 = np.asarray(gt_pcd_new.compute_point_cloud_distance(predicted_pcd_new))
            # chamfer = np.mean(d1) + np.mean(d2)

            d1 = np.asarray(sampled_predicted_pcd.compute_point_cloud_distance(sampled_gt_pcd))
            d2 = np.asarray(sampled_gt_pcd.compute_point_cloud_distance(sampled_predicted_pcd))
            chamfer = np.mean(d1) + np.mean(d2)

            # Compute Hausdorff Distance
            hausdorff = max(np.max(d1), np.max(d2))

            fscore, precision, recall = calculate_fscore(predicted_pcd, gt_pcd, 0.1)

            writer.writerow([filename, chamfer, hausdorff, fscore, precision, recall])
            print(f"{filename} -> Chamfer: {chamfer:.4f}, Hausdorff: {hausdorff:.4f}, F-Score: {fscore:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

def main():

    # evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/Taurus/extended/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/donut_3/meshes/donut1.ply")
    # evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/Taurus/OSVP/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/donut_3/meshes/donut1.ply")

    # Pitcher Base
    # crop_mesh("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/PitcherBase/extended/meshes/220.000.ply")
    evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/PitcherBase/extended4/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/PitcherBase/textured.obj")
    # evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/PitcherBase/OSVP2/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/PitcherBase/textured.obj")

    # Tissue Box
    # crop_mesh("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/TissueBox/extended/meshes/200.000.ply")
    # evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/TissueBox/extended2/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/TissueBox/meshes/box.ply")
    # evaluate("/home/luke/ros/humble/system/src/comp_vis_pkg/scripts/iSDF/results/iSDF/TissueBox/OSVP3/meshes", "/home/luke/ros/humble/system/src/franka_vp2/models/TissueBox//meshes/box.ply")

if __name__ == "__main__":
    main()
