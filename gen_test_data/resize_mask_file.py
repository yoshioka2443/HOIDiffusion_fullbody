# %%
import os
from PIL import Image

def resize_and_crop_images(root_dir, save_dir):

    # ディレクトリ内のすべてのファイルをループ
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)

        # PNGファイルのみ処理
        if os.path.isfile(file_path) and file_name.lower().endswith('.png'):
            try:
                with Image.open(file_path) as img:
                    # 画像のサイズを取得
                    width, height = img.size

                    # 中央3/4部分の範囲を計算
                    left = width * 1/8
                    right = width * 7/8
                    top = height * 1/8
                    bottom = height * 7/8

                    # 画像をクロップ
                    cropped_img = img.crop((left, top, right, bottom))

                    # クロップした画像を512x512にリサイズ
                    resized_img = cropped_img.resize((512, 512), Image.LANCZOS)

                    # 上書き保存（オリジナルを保護したい場合は別名で保存）
                    # resized_img.save(file_path)
                    resized_img.save(os.path.join(save_dir, file_name))

                    print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

# 実行

root_dir = "/home/datasets/test_data_NIMBLE"
resize_mask_path = os.path.join(root_dir, "mask_pre_resize")
mask_path = os.path.join(root_dir, "mask")
if not os.path.exists(mask_path):
    os.makedirs(mask_path)
resize_and_crop_images(resize_mask_path, mask_path)