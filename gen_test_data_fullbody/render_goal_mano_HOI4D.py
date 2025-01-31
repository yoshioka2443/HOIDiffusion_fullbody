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

from tqdm import tqdm
import csv
from pathlib import Path


# A serene beach at sunset.
# A bustling cityscape at night.
# A tranquil forest with a winding river.
# A snow-covered mountain range.
# A vibrant desert oasis.
# An idyllic countryside with rolling hills.
# A futuristic sci-fi cityscape.
# A tropical island paradise.
# A foggy, mysterious swamp.
# A rocky, barren lunar landscape.
scene_info = ["A serene beach at sunset.", "A bustling cityscape at night.", "A tranquil forest with a winding river.", "A snow-covered mountain range.", "A vibrant desert oasis.", "An idyllic countryside with rolling hills.", "A futuristic sci-fi cityscape.", "A tropical island paradise.", "A foggy, mysterious swamp.", "A rocky, barren lunar landscape."]

LIMBS = [ (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (15, 22), (15, 23), (15, 24)]
LEFT_HAND_LIMBS = [ (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36), (20, 37), (37, 38), (38, 39)]
RIGHT_HAND_LIMBS = [ (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51), (21, 52), (52, 53), (53, 54)]
LIMBS += LEFT_HAND_LIMBS
LIMBS += RIGHT_HAND_LIMBS

def depth_min_max(obj_file_path, sbj_file_path, output_size=(1024, 1024)):
    # Create PyRender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])

    sbj_mesh = trimesh.load(sbj_file_path)
    sbj_mesh_node = pyrender.Mesh.from_trimesh(sbj_mesh, smooth=False)
    sbj_node = scene.add(sbj_mesh_node, name="subject")

    obj_mesh = trimesh.load(obj_file_path)
    obj_mesh_node = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
    obj_node = scene.add(obj_mesh_node, name="object")

    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 1, 1.5]

    scene.add(camera, pose=camera_pose)

    # Render offscreen
    renderer = pyrender.OffscreenRenderer(*output_size)
    color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

    # Create depth image
    # print("min depth:", np.min(depth))
    # print("max depth:", np.max(depth))
    # depth_valid = depth[depth > 0]
    # min_depth = np.min(depth_valid)
    # max_depth = np.max(depth_valid)

    depth_valid = depth[depth > 0]
    min_depth = np.min(depth_valid)
    max_depth = np.max(depth)
    return min_depth, max_depth


def extract_mano_from_smplx(sbj_mesh, hand="right"):
    """
    Extracts the hand mesh from an SMPLX mesh
    hand: "right" or "left"
    """
    # smplx_to_mano_indices = np.load("mano_indices.npy")  # Precomputed MANO indices from SMPLX
    smplx_to_mano_indices = np.load("/home/datasets/arctic/data/body_models/mano/MANO_SMPLX_vertex_ids.pkl", allow_pickle=True)
    if hand == "right":
        mano_indices = smplx_to_mano_indices["right_hand"]
    else:
        mano_indices = smplx_to_mano_indices["left_hand"]
    
    mano_vertices = sbj_mesh.vertices[mano_indices]
    # mano_faces = sbj_mesh.faces[mano_indices]  # Faces must be filtered accordingly
    if hand == "left":
        mano_pkl = np.load("/home/datasets/arctic/data/body_models/mano/MANO_LEFT.pkl", allow_pickle=True, encoding="latin1")
    else:
        mano_pkl = np.load("/home/datasets/arctic/data/body_models/mano/MANO_RIGHT.pkl", allow_pickle=True, encoding="latin1")
    
    mano_faces = mano_pkl["f"]

    return trimesh.Trimesh(vertices=mano_vertices, faces=mano_faces, process=False)

def draw_skeleton(image, joints_2d=None):
    imgIn = np.zeros_like(image)
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = LIMBS

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
            continue
            # color_num = 4
            # color_code_num = (limb_num // color_num)
            # limb_color = [c + 35 * (limb_num % color_num) for c in joint_color_code[color_code_num]]
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

def render_skeleton(obj_file_path, sbj_file_path, output_size=(1024, 1024)):
    # Load joints and vertices data
    obj_mesh = trimesh.load(obj_file_path)
    sbj_mesh = trimesh.load(sbj_file_path)

    # Define camera position and orientation
    camera_position = np.array([0, 1, 1.5])
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
    # focal_length = 500  # Example focal length
    # image_size = (1024, 1024)
    focal_length = 443.41 * 2
    # image_size = (512, 512)
    image_size = output_size
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

    regressor = model_data['J_regressor']

    joints = regressor @ sbj_mesh.vertices
    vertices_camera = (rotation_matrix @ joints.T).T + camera_position
    vertices_2d = (intrinsic_matrix @ vertices_camera.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]
    joints_2d = vertices_2d

    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    rendered_image = draw_skeleton(image, joints_2d)

    return cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)

def generate_depth_and_mask(obj_file_path, sbj_file_path, output_size=(1024, 1024)):
    # Load meshes
    obj_mesh = trimesh.load(obj_file_path)
    sbj_mesh = trimesh.load(sbj_file_path)

    # Extract MANO hand mesh
    mano_mesh_left = extract_mano_from_smplx(sbj_mesh, "left")
    mano_mesh_right = extract_mano_from_smplx(sbj_mesh, "right")
    
    # Color setup
    # obj_mesh.visual.vertex_colors = np.array([[0, 0, 128, 255]] * len(obj_mesh.vertices))
    obj_mesh.visual.vertex_colors = np.array([[51, 50, 153, 255]] * len(obj_mesh.vertices))
    mano_mesh_left.visual.vertex_colors = np.array([[128, 128, 128, 255]] * len(mano_mesh_left.vertices))
    mano_mesh_right.visual.vertex_colors = np.array([[128, 128, 128, 255]] * len(mano_mesh_right.vertices))
    
    # Create PyRender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    
    # Add MANO hand mesh
    mano_mesh_left_node = pyrender.Mesh.from_trimesh(mano_mesh_left, smooth=False)
    scene.add(mano_mesh_left_node, name="mano_hand_left")

    mano_mesh_right_node = pyrender.Mesh.from_trimesh(mano_mesh_right, smooth=False)
    scene.add(mano_mesh_right_node, name="mano_hand_right")

    # Add object mesh
    obj_mesh_node = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
    obj_node = scene.add(obj_mesh_node, name="object")
    
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
    
    min_depth, max_depth = depth_min_max(obj_file_path, sbj_file_path, output_size)
    # print("min depth:", min_depth)
    # print("max depth:", max_depth)
    depth_normalized = (depth - min_depth) / (max_depth - min_depth)
    depth_image = np.where(depth > 0, (1 - depth_normalized) * 255, 0).astype(np.uint8)
    
    # Create mask
    mask = np.where(depth > 0, 255, 0).astype(np.uint8)
    
    return depth_image, segmentation, mask

def resize_image(img, size=(512, 512)):
    """Resize the image (NumPy array) to the desired size using manual interpolation."""
    height, width, *channels = img.shape
    resized = np.zeros((size[0], size[1], *channels), dtype=img.dtype)

    # Calculate scale factors
    scale_x = width / size[1]
    scale_y = height / size[0]

    for i in range(size[0]):
        for j in range(size[1]):
            # Map the target pixel to the source image
            src_x = int(j * scale_x)
            src_y = int(i * scale_y)

            # Assign the nearest neighbor value
            resized[i, j] = img[src_y, src_x]

    return resized

def center_crop(img, size=(512, 512)):
    """Crop the image (NumPy array) to a square by keeping the center and cutting off the longer dimension."""
    
    cropped = img.copy()
    height, width = size

    cx = img.shape[1] // 2
    cy = img.shape[0] // 2

    cropped = cropped[cy - height // 2:cy + height // 2, cx - width // 2:cx + width // 2]
    return cropped

# Main function
def main():
    output_dir = Path("/home/datasets/hand_test_abc_10scenes_HOI4D")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_num = 1
    paths_info = []
    prompts = ["A person is grasping something."]
    ply_root_dir = Path("/home/datasets/GOAL_outputs/static")

    object_list = [obj.name for obj in ply_root_dir.iterdir() if obj.is_dir() and obj.name != "outputs"]
    print(object_list)

    # 必要な出力フォルダを事前に作成
    sub_dirs = ["depth", "mask", "seg", "skeleton", "image"]
    for sub_dir in sub_dirs:
        (output_dir / sub_dir).mkdir(exist_ok=True)
   

    for sidx in tqdm(range(len(scene_info))):
        for object_file_name in tqdm(object_list):
            for i in tqdm(range(scene_num)):
                obj_file_path = os.path.join(ply_root_dir, object_file_name, f"{i:04d}_obj.ply")
                sbj_file_path = os.path.join(ply_root_dir, object_file_name, f"{i:04d}_sbj_refine.ply")
                file_name = f"{object_file_name}_{i:04d}"
                # object_name = obj_file_path.split("/")[-1].split("_")[1]
                object_name = object_file_name.split("_")[1]
                print(object_name)
                depth, mask, seg = generate_depth_and_mask(obj_file_path, sbj_file_path)
                skeleton = render_skeleton(obj_file_path, sbj_file_path)

                mask = center_crop(mask, (512, 512))
                seg = center_crop(seg, (512, 512))
                depth = center_crop(depth, (512, 512))
                skeleton = center_crop(skeleton, (512, 512))

                mask = resize_image(mask, (512, 512))
                seg = resize_image(seg, (512, 512))
                depth = resize_image(depth, (512, 512))
                skeleton = resize_image(skeleton, (512, 512))

                # 保存
                cv2.imwrite(str(output_dir / "depth" / f"{file_name}_depth_{i}_{sidx}.png"), depth)
                cv2.imwrite(str(output_dir / "mask" / f"{file_name}_mask_{i}_{sidx}.png"), mask)
                cv2.imwrite(str(output_dir / "seg" / f"{file_name}_seg_{i}_{sidx}.png"), seg)
                cv2.imwrite(str(output_dir / "skeleton" / f"{file_name}_skeleton_{i}_{sidx}.png"), skeleton)
                cv2.imwrite(str(output_dir / "image" / f"{file_name}_image_{i}_{sidx}.png"), mask)
                info = {
                    "image": str((output_dir / "image" / f"{file_name}_image_{i}_{sidx}.png").resolve()),
                    "top": 0, "bottom": 512, "left": 0, "right": 512,
                    "sentence": f"A person is grasping {object_name}. " + scene_info[i],
                    "seg": str((output_dir / "seg" / f"{file_name}_seg_{i}_{sidx}.png").resolve()),
                    "mask": str((output_dir / "mask" / f"{file_name}_mask_{i}_{sidx}.png").resolve()),
                    "depth": str((output_dir / "depth" / f"{file_name}_depth_{i}_{sidx}.png").resolve()),
                    "skeleton": str((output_dir / "skeleton" / f"{file_name}_skeleton_{i}_{sidx}.png").resolve())
                }
                paths_info.append(info)

    
    # Save paths_info to a CSV file
    csv_file_path = os.path.join(output_dir, "paths_info.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames= ["image", "top", "bottom", "left", "right", "sentence", "seg", "mask", "depth", "skeleton"])
        writer.writeheader()
        writer.writerows(paths_info)

if __name__ == "__main__":
    main()


# %%
