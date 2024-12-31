import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def resize_and_pad_image(img, target_size=512):
    # 元の画像のサイズを取得
    original_height, original_width = img.shape[:2]
    
    # スケールを計算
    scale = target_size / max(original_width, original_height)
    
    # 拡大または縮小したサイズを計算
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 画像をリサイズ
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # パディングするサイズを計算
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top_pad = delta_h // 2
    bottom_pad = delta_h - top_pad
    left_pad = delta_w // 2
    right_pad = delta_w - left_pad
    
    # 黒でパディング
    if len(img.shape) == 3:
        color = [0, 0, 0]
    else:
        color = 0
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad,
                                    cv2.BORDER_CONSTANT, value=color)
    print(padded_img.shape)
    return padded_img

# def save_images(original, render, background_image, depth_image, output_dir, image_num):
def save_images(original, render, albedo_image, background_image, depth_image, output_dir, image_num):
    """画像を保存および表示します。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_image = torch.pow(original, 1.0 / 2.2).cpu().detach().numpy()
    rendered_image = torch.pow(render, 1.0 / 2.2).cpu().detach().numpy()
    rendered_image_rgb = rendered_image[..., :3]
    albedo_image = torch.pow(albedo_image, 1.0 / 2.2).cpu().detach().numpy()

    # マスクの保存
    mask_image = rendered_image[..., -1]
    mask_image_resized = resize_and_pad_image(mask_image)

    os.makedirs(os.path.join(output_dir, "seg"), exist_ok=True)
    seg_path = os.path.join(output_dir, "seg", f'seg{image_num}.png')
    plt.imsave(seg_path, mask_image_resized, cmap='gray')

    plt.imshow(mask_image_resized, cmap='gray')
    plt.axis('off')
    # plt.savefig(seg_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # アルベド画像の保存
    albedo_image_resized = resize_and_pad_image(albedo_image)

    os.makedirs(os.path.join(output_dir, "albedo"), exist_ok=True)
    albedo_path = os.path.join(output_dir, "albedo", f'albedo_image{image_num}.png')
    plt.imsave(albedo_path, albedo_image_resized)

    plt.imshow(albedo_image_resized)
    plt.axis('off')
    # plt.savefig(albedo_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # デプス画像の保存
    depth_image_np = depth_image.cpu().numpy()
    depth_image_resized = resize_and_pad_image(depth_image_np)

    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    depth_path = os.path.join(output_dir, "depth", f'depth_image{image_num}.png')
    plt.imsave(depth_path, depth_image_resized, cmap='gray')

    plt.imshow(depth_image_resized, cmap='gray')
    plt.axis('off')
    # plt.savefig(depth_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # マスクを適用して背景を黒にする
    mask_image_3ch = np.repeat(mask_image[..., np.newaxis], 3, axis=-1) 
    masked_rendered_image = rendered_image_rgb * mask_image_3ch
    masked_rendered_image_resized = resize_and_pad_image(masked_rendered_image)

    os.makedirs(os.path.join(output_dir, "texture"), exist_ok=True)
    texture_path = os.path.join(output_dir, "texture", f'masked_rendered_image{image_num}.png')
    plt.imsave(texture_path, masked_rendered_image_resized)

    plt.imshow(masked_rendered_image_resized)
    plt.axis('off')
    # plt.savefig(texture_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # レンダリング結果の保存
    
    background_image_resized = resize_and_pad_image(background_image)
    rendered_image_resized = resize_and_pad_image(rendered_image_rgb)

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    rgb_path = os.path.join(output_dir, "rgb", f'rendered_image{image_num}.png')

    mask_image_3ch_resized = resize_and_pad_image(mask_image_3ch)
    print("rendered_image_resized.shape", rendered_image_resized.shape)
    print("mask_image_3ch.shape", mask_image_3ch_resized.shape)
    print("background_image_resized.shape", background_image_resized.shape)
    
    rendered_image_resized = rendered_image_resized * mask_image_3ch_resized + background_image_resized * (1 - mask_image_3ch_resized)

    plt.imsave(rgb_path, rendered_image_resized)
    plt.imshow(rendered_image_resized)
    plt.axis('off')
    # plt.savefig(rgb_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # パスを辞書で返す
    return {'seg': seg_path, 'rgb': rgb_path, 'depth': depth_path, 'texture': texture_path, 'albedo': albedo_path}

def display_textures(images, output_dir, image_num):
    """テクスチャ画像を表示および保存します。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    axs = [axs] if len(images) == 1 else axs
    for ax, (name, img) in zip(axs, images.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'texture_info{image_num}.png'), bbox_inches='tight', pad_inches=0)
    plt.show()