# %%
import pyrender
import numpy as np
import cv2
import os
import torch
import trimesh
# import plotly.graph_objects as go
import pickle
import math

LIMBS = [ (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (15, 22), (15, 23), (15, 24)]
LEFT_HAND_LIMBS = [ (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36), (20, 37), (37, 38), (38, 39)]
RIGHT_HAND_LIMBS = [ (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51), (21, 52), (52, 53), (53, 54)]
LIMBS += LEFT_HAND_LIMBS
LIMBS += RIGHT_HAND_LIMBS

def draw_skeleton(image, joints_2d=None):
    imgIn = np.zeros_like(image)
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = LIMBS

    for joint_num in range(joints_2d.shape[0]):
        if joint_num < 24:
            color_num = 4
            color_code_num = (joint_num // color_num)
            joint_color = [c + 35 * (joint_num % color_num) for c in joint_color_code[color_code_num]]
        elif joint_num < 39:
            color_num = 3
            color_code_num = ((joint_num - 24) // color_num)
            joint_color = [c + 35 * ((joint_num - 24) % color_num) for c in joint_color_code[color_code_num]]
        elif joint_num < 54:
            color_num = 3
            color_code_num = ((joint_num - 39) // color_num)
            joint_color = [c + 35 * ((joint_num - 39) % color_num) for c in joint_color_code[color_code_num]]
        else:
            joint_color = [255, 255, 255]
        
        try:
            x = int(joints_2d[joint_num, 0])
            y = int(joints_2d[joint_num, 1])
            cv2.circle(imgIn, center=(x, y), radius=1, color=joint_color, thickness=-1)
        except:
            print("error", joint_num)

    for limb_num in range(len(limbs)):
        x1 = joints_2d[limbs[limb_num][0], 1]
        y1 = joints_2d[limbs[limb_num][0], 0]
        x2 = joints_2d[limbs[limb_num][1], 1]
        y2 = joints_2d[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                    (int(length / 2), 1),
                                    int(deg),
                                    0, 360, 1)

        if limb_num < 24:
            color_num = 4
            color_code_num = (limb_num // color_num)
            limb_color = [c + 35 * (limb_num % color_num) for c in joint_color_code[color_code_num]]
        elif limb_num < 39:
            color_num = 3
            color_code_num = ((limb_num - 24) // color_num)
            limb_color = [c + 35 * ((limb_num - 24) % color_num) for c in joint_color_code[color_code_num]]
        elif limb_num < 54:
            color_num = 3
            color_code_num = ((limb_num - 39) // color_num)
            limb_color = [c + 35 * ((limb_num - 39) % color_num) for c in joint_color_code[color_code_num]]
        else:
            limb_color = [255, 255,255]
        cv2.fillConvexPoly(imgIn, polygon, color=limb_color)
    return imgIn

def render_skeleton(obj_file_path, sbj_file_path, root_dir):
    # Load joints and vertices data
    obj_mesh = trimesh.load(obj_file_path)
    sbj_mesh = trimesh.load(sbj_file_path)

    # Define camera position and orientation
    camera_position = np.array([0, 1, 1])
    look_at = np.array([0, 1, 0])
    up_vector = np.array([0, -1, 0])

    # Compute camera rotation matrix
    z_axis = (look_at - camera_position).astype(np.float64)  # 明示的に float64 にキャスト
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Project vertices to 2D
    focal_length = 500  # Example focal length
    # image_size = (1024, 1024)
    image_size = (512, 512)
    intrinsic_matrix = np.array([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ])

    # bodymesh
    vertices_camera = (rotation_matrix @ sbj_mesh.vertices.T).T + camera_position
    vertices_2d = (intrinsic_matrix @ vertices_camera.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]
    bodymesh_vertices_2d = vertices_2d

    # objmesh
    vertices_camera = (rotation_matrix @ obj_mesh.vertices.T).T + camera_position
    vertices_2d = (intrinsic_matrix @ vertices_camera.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]
    objmesh_vertices_2d = vertices_2d

    model_path = "/home/datasets/arctic/data/body_models/smplx/SMPLX_NEUTRAL.pkl"
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f, encoding='latin1')  # 必要に応じてエンコーディングを指定
    print(model_data.keys()) 

    regressor = model_data['J_regressor']
    print(regressor.shape)  # (24, 10475)

    joints = regressor @ sbj_mesh.vertices
    print(joints.shape)  # (24, 3) 

    vertices_camera = (rotation_matrix @ joints.T).T + camera_position
    vertices_2d = (intrinsic_matrix @ vertices_camera.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]
    joints_2d = vertices_2d

    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    rendered_image = draw_skeleton(image, joints_2d)

    # Save the image
    output_path = "output_skeleton.png"
    output_path = os.path.join(root_dir, output_path)
    cv2.imwrite(output_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
    
    print(f"Rendered image saved at {output_path}")


def generate_depth_and_mask(obj_file_path, sbj_file_path, root_dir, output_size=(512, 512)):
    # Load meshes
    obj_mesh = trimesh.load(obj_file_path)
    sbj_mesh = trimesh.load(sbj_file_path)

    obj_mesh.visual.vertex_colors = np.array([[100, 100, 100, 255]] * len(obj_mesh.vertices))
    sbj_mesh.visual.vertex_colors = np.array([[150, 150, 150, 255]] * len(sbj_mesh.vertices))

    # Create PyRender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])

    # Add object mesh
    obj_mesh_node = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
    obj_node = scene.add(obj_mesh_node, name="object")

    # Add subject (SMPLX) mesh
    sbj_mesh_node = pyrender.Mesh.from_trimesh(sbj_mesh, smooth=False)
    sbj_node = scene.add(sbj_mesh_node, name="subject")

    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 1, 1.5]

    scene.add(camera, pose=camera_pose)

    # Setup light
    light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render offscreen
    renderer = pyrender.OffscreenRenderer(*output_size)
    color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

    # Create segmentation image
    segmentation = color.copy()

    # Create depth image
    depth_valid = depth[depth > 0]
    min_depth = np.min(depth_valid)
    depth_normalized = (depth - min_depth) / (np.max(depth) - min_depth)
    depth_image = np.where(depth > 0, (1 - depth_normalized) * 255, 0).astype(np.uint8)

    # Create mask
    mask = np.where(depth > 0, 255, 0).astype(np.uint8)

    # Save results
    # cv2.imwrite("output_depth.png", depth_image)
    # cv2.imwrite("output_seg.png", mask)
    # cv2.imwrite("output_mask.png", segmentation)

    cv2.imwrite(os.path.join(root_dir, "output_depth.png"), depth_image)
    cv2.imwrite(os.path.join(root_dir, "output_seg.png"), mask)
    cv2.imwrite(os.path.join(root_dir, "output_mask.png"), segmentation)

    print("Depth and mask images have been saved!")

# Main function
def main():
    # obj_file_path = "/home/datasets/GOAL/results/apple_grasp/0000_obj.ply"
    # sbj_file_path = "/home/datasets/GOAL/results/apple_grasp/0000_sbj_refine.ply"
    root_dir = "/home/datasets/fullbody_test_goal"
    obj_file_path = "/home/datasets/GOAL/results/binoculars_grasp/0003_obj.ply"
    sbj_file_path = "/home/datasets/GOAL/results/binoculars_grasp/0003_sbj_refine.ply"

    generate_depth_and_mask(obj_file_path, sbj_file_path, root_dir)
    render_skeleton(obj_file_path, sbj_file_path, root_dir)

if __name__ == "__main__":
    main()


# %%
