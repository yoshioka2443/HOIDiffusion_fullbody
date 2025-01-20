# %%
import numpy as np
import os


file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_0000_joints_sbj_0.npz"
if not os.path.exists(file_path):
    print(f"Error: File {file_path} does not exist.")
else:
    loaded_data = np.load(file_path)
    print("File loaded successfully.")
    print(list(loaded_data["arr_0"].shape))

verts_file_path = "/home/datasets/GOAL/results/MNet_terminal/static_and_motion_1/s10_apple_eat_1_motion_0000_verts_sbj_0.npz"
if not os.path.exists(verts_file_path):
    print(f"Error: File {verts_file_path} does not exist.")
else:
    loaded_data = np.load(verts_file_path)
    print("File loaded successfully.")
    print(list(loaded_data["arr_0"].shape))
# %%
