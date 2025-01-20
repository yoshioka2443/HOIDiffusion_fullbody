import os
from PIL import Image, ImageOps


base_dir = "/home/datasets/test_data_NIMBLE"

def process_images(base_dir):
    # ベースディレクトリ内のすべてのサブディレクトリを処理
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 対象の画像ファイル拡張子
                file_path = os.path.join(root, file)
                # 画像を開く
                with Image.open(file_path) as img:
                    # サイズが640x480か確認
                    if img.size == (640, 480):
                        # 左右両端をトリミングして中央512x512にする
                        left = (img.width - 512) // 2
                        top = (img.height - 512) // 2
                        right = left + 512
                        bottom = top + 512
                        cropped_img = img.crop((left, top, right, bottom))
                        # 保存 (上書き保存)
                        cropped_img.save(file_path)
                        print(f"Processed and saved: {file_path}")
                    else:
                        print(f"Skipped (not 640x480): {file_path}")

def process_images_with_padding(base_dir):
    # ベースディレクトリ内のすべてのサブディレクトリを処理
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 対象の画像ファイル拡張子
                file_path = os.path.join(root, file)
                # 画像を開く
                with Image.open(file_path) as img:
                    # サイズが640x480か確認
                    if img.size == (640, 480):
                        # 上下にパディングを追加して512x512にする
                        padding_color = (0, 0, 0)  # パディング色 (黒)（必要なら変更）
                        vertical_padding = (512 - img.height) // 2
                        padded_img = ImageOps.expand(
                            img, border=(0, vertical_padding, 0, vertical_padding), fill=padding_color
                        )
                        # 保存 (上書き保存)
                        padded_img.save(file_path)
                        print(f"Processed and saved: {file_path}")
                    else:
                        print(f"Skipped (not 640x480): {file_path}")



# # 処理実行
# process_images(base_dir)
# 処理実行
process_images_with_padding(base_dir)


