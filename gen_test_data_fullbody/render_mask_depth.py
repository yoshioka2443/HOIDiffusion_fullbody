# %%
import pyrender
import numpy as np
import cv2
import trimesh

# # ヘッドレスレンダリングのための設定
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

def generate_depth_and_mask(obj_file_path, sbj_file_path, output_size=(512, 512)):
    # Load meshes
    obj_mesh = trimesh.load(obj_file_path)
    sbj_mesh = trimesh.load(sbj_file_path)

    # Create PyRender scene
    scene = pyrender.Scene()

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
    color, depth = renderer.render(scene)

    # Normalize depth for visualization
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_image = ((1 - depth_normalized * 255)).astype(np.uint8)

    # Create mask
    mask = np.where(depth > 0, 255, 0).astype(np.uint8)
    
    # Save results
    cv2.imwrite("output_depth.png", depth_image)
    cv2.imwrite("output_seg.png", mask)

    # Create segmentation image
    segmentation = np.zeros_like(color, dtype=np.uint8)

    # Mask for the object (based on depth)
    obj_mask = (depth > 0).astype(np.uint8)
    scene.remove_node(sbj_node)  # Remove subject mesh temporarily
    _, obj_depth = renderer.render(scene)
    
    # Mask for the subject (based on depth)
    scene.add(sbj_mesh_node)  # Add subject mesh back
    scene.remove_node(obj_node)  # Remove object mesh temporarily
    _, sbj_depth = renderer.render(scene)

    sbj_mask = (sbj_depth > obj_depth).astype(np.uint8)
    segmentation[sbj_mask == 1] = [0, 0, 255]  # Blue for subject
    obj_mask = ((obj_depth > 0) & (obj_depth > sbj_depth - 1)).astype(np.uint8)
    segmentation[obj_mask == 1] = [255, 0, 0]  # Red for object
    cv2.imwrite("output_mask.png", segmentation)

    print("Depth and mask images have been saved!")

# Main function
def main():
    obj_file_path = "/home/datasets/GOAL/results/apple_grasp/0000_obj.ply"
    sbj_file_path = "/home/datasets/GOAL/results/apple_grasp/0000_sbj_refine.ply"

    generate_depth_and_mask(obj_file_path, sbj_file_path)
    render_skeleton(obj_file_path, sbj_file_path)

if __name__ == "__main__":
    main()


# %%
