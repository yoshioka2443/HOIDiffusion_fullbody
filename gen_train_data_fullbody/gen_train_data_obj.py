# import os
# import numpy as np
# from PIL import Image
# import pandas as pd
# from tqdm import tqdm
# import re

# # 入力ディレクトリと出力ディレクトリを設定
# input_dir = "/home/datasets/train_fullbody"
# output_dir = os.path.join("/home/datasets", "fullbody_train")
# os.makedirs(output_dir, exist_ok=True)

# # 処理内容を定義
# def process_image(img, crop_coords, resize_dims):
#     """画像をトリミングし、リサイズする"""
#     img = img.crop(crop_coords)  # トリミング
#     img = img.resize(resize_dims)  # リサイズ
#     return img

# def resize_image_with_padding(img, target_dims, fill_color=(255, 255, 255)):
#     """画像を指定サイズにリサイズし、足りない部分を指定色で補完する"""
#     target_width, target_height = target_dims
#     original_width, original_height = img.size

#     aspect_ratio = original_width / original_height
#     if target_width / target_height > aspect_ratio:
#         new_height = target_height
#         new_width = int(aspect_ratio * target_height)
#     else:
#         new_width = target_width
#         new_height = int(target_width / aspect_ratio)

#     resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#     new_img = Image.new("RGB", (target_width, target_height), fill_color)
#     paste_x = (target_width - new_width) // 2
#     paste_y = (target_height - new_height) // 2
#     new_img.paste(resized_img, (paste_x, paste_y))

#     return new_img

# def process_depth(depth_array, crop_coords, resize_dims):
#     """深度マップをトリミングし、リサイズする"""
#     # depth_array = (depth_array * 65535).astype(np.uint16)
#     depth_array = (depth_array * 255).astype(np.uint8)
#     # white blacu 逆転
#     depth_array = 255 - depth_array
#     depth_img = Image.fromarray(depth_array)
#     depth_img = depth_img.crop(crop_coords)
#     depth_img = depth_img.resize(resize_dims)
#     return np.array(depth_img)

# def apply_segmentation(mask, img):
#     """セグメンテーションマスクを適用して対象領域以外を黒にする"""
#     mask_array = np.array(mask)
#     img_array = np.array(img)
#     segmented_img = np.where(mask_array[:, :, None] > 0, img_array, 0)
#     return Image.fromarray(segmented_img)

# def create_segmentation(mask):
#     """マスク画像（seg）を生成"""
#     mask_array = np.array(mask)
#     mask_array = np.mean(mask_array, axis=2)
#     seg_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
#     return Image.fromarray(seg_array)

# # 処理パラメータ
# target_dims = (1600, 900)  # (width, height)
# crop_coords = (480, 270, 1120, 630)  # (left, top, right, bottom)
# resize_dims = (1920, 1080)  # (width, height)

# categories = ["depth", "rgb", "mask", "bodymesh", "skeleton"]
# csv_data = {"image": [], "depth": [], "mask": [], "bodymesh": [], "skeleton": [], "seg": [], "texture": [], "top": [], "bottom": [], "left": [], "right": [], "sentence": []}

# frame_num = 783

# for category in categories:
#     input_category_dir = os.path.join(input_dir, category)
#     output_category_dir = os.path.join(output_dir, category if category != "rgb" else "image")
#     os.makedirs(output_category_dir, exist_ok=True)

#     # sample
#     files = sorted(os.listdir(input_category_dir))[:frame_num]

#     for file_name in tqdm(files, desc=f"Processing {category}"):
#         input_path = os.path.join(input_category_dir, file_name)
#         output_path = os.path.join(output_category_dir, file_name)

#         if category in ["rgb", "mask", "bodymesh", "skeleton"]:
#             img = Image.open(input_path)

#             if category == "rgb":
#                 img = resize_image_with_padding(img, target_dims)
#                 processed_img = process_image(img, crop_coords, resize_dims)
#                 output_path = output_path.replace(".jpg", ".png")
#                 output_path = re.sub(r'(\d+)', lambda x: f"{int(x.group(1)) - 1:04d}", output_path)

#                 processed_img.save(output_path, "PNG")
#                 csv_data["image"].append(output_path)
            
#             elif category == "skeleton":
#                 img = resize_image_with_padding(img, target_dims)
#                 processed_img = process_image(img, crop_coords, resize_dims)
#                 processed_img.save(output_path)
#                 csv_data[category].append(output_path)

#             else:
#                 processed_img = process_image(img, crop_coords, resize_dims)
#                 processed_img.save(output_path)
#                 csv_data[category].append(output_path)

#         elif category == "depth":
#             depth_array = np.load(input_path)
#             processed_depth = process_depth(depth_array, crop_coords, resize_dims)
#             np.save(output_path, processed_depth)
#             csv_data[category].append(output_path)

# # seg と texture 処理
# mask_dir = os.path.join(output_dir, "mask")
# rgb_dir = os.path.join(output_dir, "image")
# seg_dir = os.path.join(output_dir, "seg")
# texture_dir = os.path.join(output_dir, "texture")
# os.makedirs(seg_dir, exist_ok=True)
# os.makedirs(texture_dir, exist_ok=True)

# # sample
# mask_files = sorted(os.listdir(mask_dir))[:frame_num]

# for file_name in tqdm(mask_files, desc="Processing seg and texture"):
#     mask_path = os.path.join(mask_dir, file_name)
#     rgb_path = os.path.join(rgb_dir, file_name)
#     seg_path = os.path.join(seg_dir, file_name)
#     texture_path = os.path.join(texture_dir, file_name)

#     mask = Image.open(mask_path)
#     seg = create_segmentation(mask)
#     seg.save(seg_path)
#     csv_data["seg"].append(seg_path)

#     rgb = Image.open(rgb_path)
#     texture = apply_segmentation(seg, rgb)
#     texture.save(texture_path)
#     csv_data["texture"].append(texture_path)

#     # 画像のトリミング範囲を記録
#     csv_data["top"].append(0)
#     csv_data["bottom"].append(360)
#     csv_data["left"].append(0)
#     csv_data["right"].append(640)

#     # 文章を記録
#     csv_data["sentence"].append("Human is grasping a object")

# # 統合したCSVデータを保存
# csv_file = os.path.join(output_dir, "output_paths.csv")
# df = pd.DataFrame(csv_data)
# df.to_csv(csv_file, index=False)

# print("Processing completed and consolidated CSV created.")

# %%
import os
import pathlib
# 処理内容を定義
def main():
    input_dir = "/home/datasets/arctic/render_out"
    output_dir = os.path.join("/home/datasets", "fullbody_train_goal")
    os.makedirs(output_dir, exist_ok=True)

    frame_seaquece = 100

    image_num = 0

    categories = ["rgb", "depth", "mask"]
    # csv_data = {"image": [], "depth": [], "mask": [], "skeleton": [], "seg": [], "texture": [], "top": [], "bottom": [], "left": [], "right": [], "sentence": []}

    scene_list = os.listdir(input_dir)
    for scene in scene_list:
        scene_dir = os.path.join(input_dir, scene)
        for category in categories:
            input_category_dir = os.path.join(scene_dir, "images", category)
            # output_category_dir = os.path.join(output_dir, category if category != "rgb" else "image")
            # os.makedirs(output_category_dir, exist_ok=True)
            print(input_category_dir)
            frame_list = os.listdir(input_category_dir)
            num = 0
            for frame in frame_list:
                if num % frame_seaquece != 0:
                    num += 1
                    continue
                else:
                    num += 1
                    input_path = os.path.join(input_category_dir, frame)
                    print(input_path)
                    image_num += 1
    print(image_num)









if __name__ == "__main__":
    main()

# %%
