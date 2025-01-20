import os
import random
import shutil
import numpy as np
import csv
from PIL import Image

def center_crop_resize(image_path, output_path, size=(512, 512)):
    """Crop horizontally to make the image square, then resize to the desired size."""
    with Image.open(image_path) as img:
        width, height = img.size

        # Ensure vertical dimension is the limiting factor
        if width > height:
            left = (width - height) // 2
            right = left + height
            top = 0
            bottom = height
        else:
            top = (height - width) // 2
            bottom = top + width
            left = 0
            right = width
        print("left, top, right, bottom", left, top, right, bottom)

        # Crop to square
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize(size, Image.LANCZOS)
        resized.save(output_path)

# def process_depth_file(npy_path, output_path, size=(512, 512)):
#     """Convert .npy depth file to PNG format, crop, and resize."""
#     depth_array = np.load(npy_path)
#     print(depth_array.shape)
#     depth_image = Image.fromarray((depth_array / np.max(depth_array) * 255).astype(np.uint8))

#     # Convert numpy array to square dimensions
#     height, width = depth_array.shape
#     if width > height:
#         left = (width - height) // 2
#         right = left + height
#         top = 0
#         bottom = height
#     else:
#         top = (height - width) // 2
#         bottom = top + width
#         left = 0
#         right = width

#     cropped_array = depth_array[top:bottom, left:right]
#     cropped_image = Image.fromarray((cropped_array / np.max(cropped_array) * 255).astype(np.uint8))

#     # Resize
#     resized = cropped_image.resize(size, Image.LANCZOS)
#     resized.save(output_path)

def process_depth_file(npy_path, output_path, size=(512, 512)):
    """Convert .npy depth file to PNG format, crop, and resize."""
    depth_array = np.load(npy_path)
    # print(depth_array.shape)
    # depth_image = Image.fromarray((depth_array / np.max(depth_array) * 255).astype(np.uint8))
    img = Image.fromarray(depth_array.astype(np.uint8))
    width, height = img.size

    # Ensure vertical dimension is the limiting factor
    if width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width

    # Crop to square
    cropped = img.crop((left, top, right, bottom))
    print(cropped.size)
    resized = cropped.resize(size, Image.LANCZOS)
    print(resized.size)
    resized.save(output_path)



def create_test_set(train_dir, test_dir, num_images):
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)

    # Get all subdirectories from the train directory
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(subdirs)
    # Assume all subdirectories have the same set of files
    image_files = os.listdir(os.path.join(train_dir, subdirs[1]))
    image_files = [f for f in image_files if f.endswith('.png') or f.endswith('.npy')]

    # Get random sample of file indices
    selected_files = random.sample(image_files, num_images)

    paths_info = []

    for subdir in subdirs:
        train_subdir_path = os.path.join(train_dir, subdir)
        test_subdir_path = os.path.join(test_dir, subdir)

        # Create the corresponding subdirectory in the test directory
        os.makedirs(test_subdir_path, exist_ok=True)

        # Copy and process the selected files
        for file_name in selected_files:
            if subdir == 'depth':
                file_name = f"{int(file_name.split('.')[0]):04d}.npy"
            src_file = os.path.join(train_subdir_path, file_name)
            dst_file = os.path.join(test_subdir_path, file_name)

            # print(file_name)

            if file_name.endswith('.png'):
                if subdir == 'image':
                    # src_file = os.path.join(train_subdir_path, f"{int(file_name.split('.')[0]):05d}.png")
                    src_file = os.path.join(train_subdir_path, f"{int(file_name.split('.')[0]):04d}.png")
                center_crop_resize(src_file, dst_file)
            elif file_name.endswith('.npy') and subdir == 'depth':
                dst_file = dst_file.replace('.npy', '.png')
                process_depth_file(src_file, dst_file)

            # Add paths to the info dictionary
            info = {
                "image": os.path.abspath(os.path.join(test_dir, "image", file_name.replace('.npy', '.png'))),
                "bodymesh": os.path.abspath(os.path.join(test_dir, "bodymesh", file_name.replace('.npy', '.png'))),
                "skeleton": os.path.abspath(os.path.join(test_dir, "skeleton", file_name.replace('.npy', '.png'))),
                "top": 0,
                "bottom": 512,
                "left": 0,
                "right": 512,
                "sentence": "A person is grasping something.",
                "seg": os.path.abspath(os.path.join(test_dir, "seg", file_name.replace('.npy', '.png'))),
                "mask": os.path.abspath(os.path.join(test_dir, "mask", file_name.replace('.npy', '.png'))),
                "depth": os.path.abspath(os.path.join(test_dir, "depth", file_name.replace('.npy', '.png'))),
                "texture": os.path.abspath(os.path.join(test_dir, "texture", file_name.replace('.npy', '.png'))
            }
            paths_info.append(info)

    return paths_info

# Paths
# train_dir = "fullbody_train"
# test_dir = "fullbody_test"

train_dir = "/home/datasets/fullbody_train"
test_dir = "/home/datasets/fullbody_test2"
os.makedirs(test_dir, exist_ok=True)

num_images = 10  # Adjust the number of images to select

# Run the function
paths_info = create_test_set(train_dir, test_dir, num_images)

# Save paths_info to a CSV file
csv_file_path = "paths_info.csv"
csv_file_path = os.path.join(test_dir, csv_file_path)

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["image", "bodymesh", "skeleton", "top", "bottom", "left", "right", "sentence", "seg", "mask", "depth", "texture"])
    writer.writeheader()
    writer.writerows(paths_info)

# Optionally print or save the paths info
# for info in paths_info:
#     print(info)