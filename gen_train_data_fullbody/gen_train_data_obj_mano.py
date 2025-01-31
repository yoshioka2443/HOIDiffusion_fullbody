# %%
import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import pathlib




# 処理内容を定義
def original_image_to_crop(img, original_dims):
    """
    指定された画像から余白を取り除き、元の画像のアスペクト比にリサイズする
    
    Parameters:
        img (PIL.Image.Image): パディングが追加された画像
        original_dims (tuple): 元の画像のサイズ (幅, 高さ)
    
    Returns:
        PIL.Image.Image: 余白が取り除かれ、リサイズされた画像
    """
    target_width, target_height = img.size
    original_width, original_height = original_dims

    # 元のアスペクト比を計算
    aspect_ratio = original_width / original_height
    
    # 現在のアスペクト比に基づいて余白の範囲を計算
    if target_width / target_height > aspect_ratio:
        # 縦方向に合わせる
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
        crop_x = (target_width - new_width) // 2
        crop_y = 0
    else:
        # 横方向に合わせる
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        crop_x = 0
        crop_y = (target_height - new_height) // 2

    # 余白をトリミング
    cropped_img = img.crop((crop_x, crop_y, crop_x + new_width, crop_y + new_height))

    # 元のサイズにリサイズ
    resized_img = cropped_img.resize((original_width, original_height), Image.Resampling.LANCZOS)

    return resized_img

def process_depth(depth_array):
    """深度マップをトリミングし、リサイズする"""
    # depth_array = (depth_array * 65535).astype(np.uint16)
    depth_array = (depth_array * 255).astype(np.uint8)
    # white blacu 逆転
    depth_array = 255 - depth_array
    depth_img = Image.fromarray(depth_array)
    return depth_img

def edit_mask_color(mask_image):
    """マスク画像の色を変更する"""
    # obj_target_color = [100, 100, 100]
    # obj_replacement_color = [128, 0, 0]
    # sbj_target_color = [150, 150, 150]
    # sbj_replacement_color = [128, 128, 128]
    obj_target_color = (100, 100, 100)
    obj_replacement_color = (128, 0, 0)
    # sbj_target_color = (150, 150, 150)
    sbj_target_color1= (250, 250, 250) # right hand
    sbj_target_color2 = (200, 200, 200) # left hand
    sbj_replacement_color = (128, 128, 128)
    edited_image = mask_image.copy()
    edited_image = edit_color(edited_image, obj_target_color, obj_replacement_color)
    edited_image = edit_color(edited_image, sbj_target_color1, sbj_replacement_color)
    edited_image = edit_color(edited_image, sbj_target_color2, sbj_replacement_color)
    return edited_image

def edit_color(img, target_color, replacement_color):
    # 指定した色に変更
    image = img.copy()
    pixels = image.load()
    for y in range(img.size[1]):  # 高さ
        for x in range(img.size[0]):  # 幅
            if pixels[x, y] == target_color:
                pixels[x, y] = replacement_color  # 色を変更
    return image


def apply_segmentation(mask, img):
    """セグメンテーションマスクを適用して対象領域以外を黒にする"""
    mask_array = np.array(mask)
    img_array = np.array(img)
    segmented_img = np.where(mask_array[:, :, None] > 0, img_array, 0)
    return Image.fromarray(segmented_img)

def create_segmentation(mask):
    """マスク画像（seg）を生成"""
    mask_array = np.array(mask)
    mask_array = np.mean(mask_array, axis=2)
    seg_array = np.where(mask_array > 10, 255, 0).astype(np.uint8)
    return Image.fromarray(seg_array)

def adapt_bbox(im, bbox, cap_dim=1000):
    cx, cy, dim = bbox
    dim *= 300
    im_cropped = im.crop((cx - dim / 2, cy - dim / 2, cx + dim / 2, cy + dim / 2))
    im_cropped_cap = im_cropped.resize((cap_dim, cap_dim))
    return im_cropped_cap

def reverse_adapt_bbox(cropped_img, bbox, original_img_dim):
    """
    adapt_bbox の逆処理を行い、クロップ部分を元画像から復元し、それ以外を黒で補完。
    cropped_img: adapt_bbox の出力（リサイズ後の画像）
    bbox: バウンディングボックス情報 (cx, cy, dim)
    original_img_dim: 元の画像サイズ (幅, 高さ)
    cap_dim: adapt_bbox でリサイズされたサイズ (デフォルト1000x1000)
    """
    cx, cy, dim = bbox
    dim *= 300  # adapt_bbox のスケール処理を逆算

    original_width, original_height = original_img_dim
    result_img = Image.new("RGB", (original_width, original_height), (0, 0, 0))  # 黒背景

    # クロップ領域の座標計算
    crop_x1 = cx - dim / 2
    crop_y1 = cy - dim / 2
    crop_x2 = cx + dim / 2
    crop_y2 = cy + dim / 2

    # クロップ画像を元スケールに戻す
    resized_cropped_img = cropped_img.resize((int(dim), int(dim)))

    # 貼り付ける範囲を計算
    paste_x1 = max(0, int(crop_x1))
    paste_y1 = max(0, int(crop_y1))
    paste_x2 = min(original_width, int(crop_x2))
    paste_y2 = min(original_height, int(crop_y2))

    # 貼り付ける部分だけ切り取る
    box = (
        max(0, -int(crop_x1)),  # cropped_img 内の切り取る位置
        max(0, -int(crop_y1)),
        max(0, -int(crop_x1)) + (paste_x2 - paste_x1),
        max(0, -int(crop_y1)) + (paste_y2 - paste_y1),
    )
    result_img.paste(resized_cropped_img.crop(box), (paste_x1, paste_y1))

    return result_img


def visualize_comparison(original_img, restored_img):
    # 画像を並べて表示
    plt.figure(figsize=(12, 6))

    # 元画像
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # 復元画像
    plt.subplot(1, 2, 2)
    plt.imshow(restored_img)
    plt.title("Restored Image")
    plt.axis("off")

    # 表示
    plt.show()

    # 画素数や寸法情報を出力
    print("Original Image size (width x height):", original_img.size)
    print("Restored Image size (width x height):", restored_img.size)

# 処理内容を定義
def main():
    input_dir = "/home/datasets/arctic/render_out_mano50"
    skeleton_input_dir = "/home/datasets/train_hand_goal_skeleton_10"
    output_dir = os.path.join("/home/datasets", "hand_train_50_edit_maskcolor")
    os.makedirs(output_dir, exist_ok=True)

    frame_sequence = 50
    image_num = 0

    # 処理パラメータ
    target_dims = (1600, 900)  # (width, height)
    crop_coords = (480, 270, 1120, 630)  # (left, top, right, bottom)
    resize_dims = (1920, 1080)  # (width, height)

    categories = ["rgb", "depth", "mask"]
    gen_categories = ["image", "skeleton", "seg", "texture"]
    csv_data = {"bodymesh": [], "depth": [], "mask": [], "top": [], "bottom": [], "left": [], "right": [], "sentence": [], "image": [], "skeleton": [], "seg": [], "texture": []}

    scene_list = os.listdir(input_dir)
    for scene in tqdm(scene_list):
        # scene parameter
        parts = scene.split("_")
        scene_name, object_name, action, object_num = parts[:4]
        if parts[4]=="retake":
            retake = parts[4]
            scene_object_name = parts[1:5]
        elif parts[4]=="retake2":
            retake = parts[4]
            scene_object_name = parts[1:5]
        else:
            retake = None
            scene_object_name = parts[1:4]
        camera_name = parts[4] if (retake == None) else parts[5]
        camera_idx = int(camera_name)

        # if scene_name != "s06":
        #     continue
        # if object_name != "ketchup" and object_name != "waffleiron":
        #     continue

        bbox_path = os.path.join("/home/datasets/arctic/outputs/processed_verts/seqs", scene_name, f"{'_'.join(scene_object_name)}.npy")
        # print(bbox_path)
        data = np.load(bbox_path, allow_pickle=True).item()
        bbox = data["bbox"]
        # process_image
        scene_dir = os.path.join(input_dir, scene)
        input_dir_for_frame = os.path.join(scene_dir, "images", "rgb")
        frame_list = os.listdir(input_dir_for_frame)
        # 処理対象のフレームを事前にフィルタリング
        filtered_frames = [
            frame for frame in frame_list if ((int(frame.split(".")[0]) % frame_sequence == 0) & (int(frame.split(".")[0]) != 0))
        ]
        filtered_frames.sort(key=lambda x: int(x.split(".")[0]))  # ソートして順序を保持
        for frame in tqdm(filtered_frames):
            frame_idx = int(frame.split(".")[0])
            # if frame_idx % frame_sequence == 0:
            bbox_for_frame = bbox[frame_idx][camera_idx]

            for category in categories:
                input_category_dir = os.path.join(scene_dir, "images", category)
                if category == "rgb":
                    category = "bodymesh"
                input_path = os.path.join(input_category_dir, frame)
                output_category_dir = os.path.join(output_dir, category)
                output_file_name = f"{scene}_{frame}"
                output_path = os.path.join(output_category_dir, output_file_name)
                os.makedirs(output_category_dir, exist_ok=True)

                if category == "depth":
                    input_path = input_path.replace("png", "npy")
                    output_path = output_path.replace("png", "npy")
                    depth_array = np.load(input_path)

                    img = process_depth(depth_array)
                    processed_img = original_image_to_crop(img, (2000, 2800))
                    adapt_bbox_image = adapt_bbox(processed_img, bbox_for_frame)
                    processed_depth = np.array(adapt_bbox_image)
                    np.save(output_path, processed_depth)
                elif category == "mask":
                    img = Image.open(input_path)
                    edited_img = edit_mask_color(img)
                    processed_img = original_image_to_crop(edited_img, (2000, 2800))
                    adapt_bbox_image = adapt_bbox(processed_img, bbox_for_frame)
                    adapt_bbox_image.save(output_path)
                else:
                    img = Image.open(input_path)
                    processed_img = original_image_to_crop(img, (2000, 2800))
                    adapt_bbox_image = adapt_bbox(processed_img, bbox_for_frame)
                    adapt_bbox_image.save(output_path)
                csv_data[category].append(output_path)
            for category in gen_categories:
                output_category_dir = os.path.join(output_dir, category)
                os.makedirs(output_category_dir, exist_ok=True)
                output_file_name = f"{scene}_{frame}"
                output_path = os.path.join(output_category_dir, output_file_name)

                if category == "image":
                    croped_root_dir = "/home/datasets/arctic/data/arctic_data/data/cropped_images"
                    input_path = os.path.join(croped_root_dir, scene_name, "_".join(scene_object_name), camera_name, f"{frame_idx + 1:05d}.jpg")

                    # 元画像を読み込む
                    bbox_image = Image.open(input_path)
                    if bbox_image.size != (1000, 1000):
                        print("size error")
                        print(bbox_image.size)
                    bbox_image.save(output_path)
                    
                    # adapt_bbox の逆処理を行い復元
                    original_size = (2000, 2800)
                    restored_img = reverse_adapt_bbox(bbox_image, bbox_for_frame, original_size)
                    adapt_bbox_image = adapt_bbox(restored_img, bbox_for_frame)
                    # adapt_bbox_image.save(output_path)
                    csv_data[category].append(output_path)
                    # 元画像と復元画像を並べて可視化
                    # visualize_comparison(bbox_image, adapt_bbox_image)
                elif category == "skeleton":
                    input_path = os.path.join(skeleton_input_dir, scene, frame)
                    img = Image.open(input_path)
                    processed_img = img
                    adapt_bbox_image = adapt_bbox(processed_img, bbox_for_frame)
                    # print("skeleton", adapt_bbox_image.size)
                    adapt_bbox_image.save(output_path)
                    csv_data[category].append(output_path)
                elif category == "seg":
                    output_category_dir = os.path.join(output_dir, category)

                    mask_path = os.path.join(output_dir, "mask", f"{scene}_{frame}")
                    mask = Image.open(mask_path)
                    seg = create_segmentation(mask)

                    seg.save(output_path)
                    csv_data[category].append(output_path)
                elif category == "texture":
                    seg_path = os.path.join(output_dir, "seg", f"{scene}_{frame}")
                    seg = Image.open(seg_path)
                    rgb_path = os.path.join(output_dir, "image", f"{scene}_{frame}")
                    rgb = Image.open(rgb_path)
                    texture = apply_segmentation(seg, rgb)

                    texture.save(output_path)
                    csv_data[category].append(output_path)
                else:
                    csv_data[category].append(None)

            # 画像のトリミング範囲を記録
            csv_data["top"].append(0)
            csv_data["bottom"].append(1000)
            csv_data["left"].append(0)
            csv_data["right"].append(1000)
            # 文章を記録
            csv_data["sentence"].append("Human is grasping a object")
            image_num += 1    
            # else:
            #     pass
        #     if image_num > 8:
        #         break
        # if image_num > 8:
        #     break

    # 統合したCSVデータを保存
    csv_file = os.path.join(output_dir, "output_paths.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print("image_num", image_num)



if __name__ == "__main__":
    main()

