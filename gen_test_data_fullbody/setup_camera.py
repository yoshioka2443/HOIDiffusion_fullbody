# %%
import os
import numpy as np
import torch
import trimesh
import cv2
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

def plot_mesh(vertices, obj_vertices, camera_position, obj_vertices2):
    # Visualize using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=0.3, color='blue')
    ))

    # Add camera position
    fig.add_trace(go.Scatter3d(
        x=[camera_position[0]],
        y=[camera_position[1]],
        z=[camera_position[2]],
        mode='markers',
        marker=dict(size=5, color='red')
    ))
    # obj
    fig.add_trace(go.Scatter3d(
        x=obj_vertices[:, 0],
        y=obj_vertices[:, 1],
        z=obj_vertices[:, 2],
        mode='markers',
        marker=dict(size=0.3, color='green')
    ))

    fig.add_trace(go.Scatter3d(
        x=obj_vertices2[:, 0],
        y=obj_vertices2[:, 1],
        z=obj_vertices2[:, 2],
        mode='markers',
        marker=dict(size=0.3, color='yellow')
    ))

    fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=1, z=1))))
    fig.show()


def setup_smplx_scene(joints_file_path, verts_file_path, obj_file_path):
    # Check if files exist
    if not os.path.exists(joints_file_path):
        print(f"Error: File {joints_file_path} does not exist.")
        return
    if not os.path.exists(verts_file_path):
        print(f"Error: File {verts_file_path} does not exist.")
        return

    # Load joints and vertices data
    joints_data = np.load(joints_file_path)
    verts_data = np.load(verts_file_path)
    obj_mesh = trimesh.load(obj_file_path)

    # Extract vertices and joint positions
    vertices = verts_data["arr_0"]
    joints = joints_data["arr_0"]

    print("Vertices shape:", vertices.shape)
    print("Joints shape:", joints.shape)

    # Normalize SMPLX model position
    # foot_position = joints[0]  # Assuming the foot is at the first joint
    # vertices -= foot_position  # Translate so the foot is at (0, 0, 0)
    # joints -= foot_position

    # Adjust model center to (0, 1, 0)
    # center_offset = np.array([0, 1, 0]) - joints.mean(axis=0)
    # vertices += center_offset
    # joints += center_offset

    # Define camera position and orientation
    camera_position = np.array([0, 1, 1])
    look_at = np.array([0, 1, 0])
    up_vector = np.array([0, -1, 0])

    # add obj
    obj_mesh2 = trimesh.load("/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_meshes/0100_obj.ply")
    # Plot the mesh
    plot_mesh(vertices, obj_mesh.vertices, camera_position, obj_mesh2.vertices)

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

    vertices_camera = (rotation_matrix @ vertices.T).T + camera_position
    vertices_2d = (intrinsic_matrix @ vertices_camera.T).T
    vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 2:3]


    # Create an image and draw the model
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    for vertex in vertices_2d:
        x, y = int(vertex[0]), int(vertex[1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    joint_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
    ]

    for i, joint in enumerate(joints):
        x, y = int(vertices_2d[i][0]), int(vertices_2d[i][1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            cv2.circle(image, (x, y), 2, joint_colors[i], -1)

    # obj
    # obj_mesh.vertices -= foot_position
    # obj_mesh.vertices -= center_offset
    # obj_mesh.vertices = (rotation_matrix @ obj_mesh.vertices.T).T + camera_position
    # obj_mesh.vertices_2d = (intrinsic_matrix @ obj_mesh.vertices.T).T
    # obj_mesh.vertices_2d = obj_mesh.vertices_2d[:, :2] / obj_mesh.vertices_2d[:, 2:3]

    # for vertex in obj_mesh.vertices_2d:
    #     x, y = int(vertex[0]), int(vertex[1])
    #     if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
    #         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)



    # Save the image
    output_path = "smplx_render.png"
    cv2.imwrite(output_path, image)
    print(f"Rendered image saved at {output_path}")


def main():
    # File paths
    joints_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_0000_joints_sbj_0.npz"
    verts_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_0000_verts_sbj_0.npz"
    obj_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_meshes/0000_obj.ply"
    
    # joints_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_static_0000_joints_sbj_0.npz"
    # verts_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_static_0000_verts_sbj_0.npz"
    # obj_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_meshes/0000_obj.ply"

    # obj_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s1_apple_eat_1_static_meshes/0000_obj.ply"

    obj_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_meshes/0100_sbj.ply"
    # obj_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_meshes/0100_obj.ply"

    # Run the function
    setup_smplx_scene(joints_file_path, verts_file_path, obj_file_path)
    # render_skeleton(joints_file_path)
    # render_depth(vertices_file_path, obj_file_path)
    # render_mask(vertices_file_path, obj_file_path)

if __name__ == "__main__":
    main()
