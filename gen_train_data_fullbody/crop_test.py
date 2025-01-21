# %%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def compare_images(gt_path, output_path):
    # GTとoutputの画像を読み込む
    gt_image = Image.open(gt_path).convert("RGB")
    output_image = Image.open(output_path).convert("RGB")
    
    # 可視化
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt_image)
    ax[0].set_title("GT Image")
    ax[0].axis("off")
    
    ax[1].imshow(output_image)
    ax[1].set_title("Output Image")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # 画素数や寸法情報を出力
    print("GT Image size (width x height):", gt_image.size)
    print("Output Image size (width x height):", output_image.size)


def adapt_bbox(im, bbox, cap_dim=1000):
    cx, cy, dim = bbox
    dim *= 300
    im_cropped = im.crop((cx - dim / 2, cy - dim / 2, cx + dim / 2, cy + dim / 2))
    im_cropped_cap = im_cropped.resize((cap_dim, cap_dim))
    return im_cropped_cap

def reverse_adapt_bbox(cropped_img, bbox, original_img_dim, cap_dim=1000):
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


# main 処理
def main():
    # パスの指定
    frame_num = 50
    camera_num = 1
    path = "/home/datasets/arctic/outputs/processed_verts/seqs/s01/box_use_01.npy"
    input_path = "/home/datasets/arctic/data/arctic_data/data/images/s01/box_grab_01/1/00050.jpg"
    output_path = "/home/datasets/fullbody_train_goal/test_crop.png"
    GT_path = "/home/datasets/arctic/data/arctic_data/data/cropped_images/s01/box_grab_01/1/00050.jpg"
    restored_output_path = "/home/datasets/fullbody_train_goal/restored_image.png"
    comparison_output_path = "/home/datasets/fullbody_train_goal/comparison_image.png"

    # bbox データを読み込む
    data = np.load(path, allow_pickle=True).item()
    bbox = data["bbox"]
    bbox_for_frame = bbox[frame_num][camera_num]

    # 元画像を読み込む
    img = Image.open(input_path)
    original_size = img.size

    # adapt_bbox によるクロップ処理
    bbox_image = adapt_bbox(img, bbox_for_frame)
    bbox_image.save(output_path)
    compare_images(GT_path, output_path)

    # adapt_bbox の逆処理を行い復元
    restored_img = reverse_adapt_bbox(bbox_image, bbox_for_frame, original_size)
    restored_img.save(restored_output_path)

    # 元画像と復元画像を並べて可視化
    visualize_comparison(img, restored_img)


if __name__ == "__main__":
    main()

