# %%
# 標準ライブラリのインポート
import os
import sys
import pickle
import math
import csv  # 追加

# サードパーティライブラリのインポート
import numpy as np
import torch
import cv2
import plotly.graph_objects as go

# カスタムモジュールのインポート
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)
print(os.getcwd())

# %%
from gen_test_data.image_edit import display_textures
from gen_test_data.runner import Runner
# %%
def main():
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("No GPU detected.")

    output_dir = 'sample'
    
    runner = Runner()
    # load handmesh, obj mesh, camera parameters
    runner.load_data(sequence_name='GPMF12', frame_number=250, replacement_object_name='010_potted_meat_can')
    runner.make_original_scene()
    runner.render_original_scene()
    runner.estimate_envmap()
    runner.render_scene()
    runner.save_images(output_dir)

    """
    デバイスの指定
    テクスチャの最適化
    replace

    """

    # # テクスチャの投影と最適化
    # projected_texture = runner.project_texture_to_uv(runner.background_image)
    # mask = runner.compute_front_mask()
    # projected_texture_masked = projected_texture * mask + torch.ones_like(projected_texture) * (1 - mask)
    # nimble_texture, _ = runner.optimize_texture(
    #     projected_texture_masked, mask, n_iterations=100, learning_rate=1e-5, lambda_reg_tex=1e3)

    # # テクスチャのブレンド
    # blended_texture = mask * projected_texture + (1 - mask) * nimble_texture

    # # オブジェクトを置き換え、手とオブジェクトの頂点を取得
    # hand_vertices, object_vertices = runner.replace_object()

    # # シーンをレンダリング
    # rendered_blended, render_albedo = runner.render_scene(hand_vertices, object_vertices)
    # original_render = runner.render_original_scene()
    # depth_image = runner.render_depth(hand_vertices, object_vertices)

    # # スケルトンの描画
    # skeleton_path = runner.render_skeleton(0)

    # # 画像を保存し、パスを取得
    # image_paths = save_images(original_render, rendered_blended, render_albedo, runner.background_image, depth_image, output_dir, str(0))

    # # パスを収集するリストを初期化
    # paths_list = []

    # # パスをリストに追加
    # paths_list.append({
    #     'seg': image_paths['seg'],
    #     'image': image_paths['rgb'],
    #     'depth': image_paths['depth'],
    #     'texture': image_paths['texture'],
    #     'albedo': image_paths['albedo'],
    #     'skeleton': skeleton_path
    # })

    # # テクスチャの表示
    # textures = {
    #     'UV Projected': projected_texture.cpu().numpy(),
    #     'UV Masked': projected_texture_masked.cpu().numpy(),
    #     'NIMBLE Optimized': nimble_texture.detach().cpu().numpy(),
    #     'Blended Texture': blended_texture.detach().cpu().numpy(),
    # }
    # display_textures(textures, output_dir, str(0))

    # # 複数のオブジェクトに対する処理
    # replacement_object_name_list = [
    #     '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', 
    #     '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', 
    #     '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser', 
    #     '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', 
    #     '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
    # ]

    # for i, object_name in enumerate(replacement_object_name_list, start=1):
    #     if i > 1:
    #         break
    #     runner.load_data(sequence_name='GPMF12', frame_number=250, replacement_object_name=object_name)
    #     hand_vertices, object_vertices = runner.replace_object()

    #     # # テクスチャの投影と最適化
    #     # projected_texture = runner.project_texture_to_uv(runner.background_image)
    #     # mask = runner.compute_front_mask()
    #     # projected_texture_masked = projected_texture * mask + torch.ones_like(projected_texture) * (1 - mask)
    #     # nimble_texture, _ = runner.optimize_texture(
    #     #     projected_texture_masked, mask, n_iterations=100, learning_rate=1e-5, lambda_reg_tex=1e3)

    #     # # テクスチャのブレンド
    #     # blended_texture = mask * projected_texture + (1 - mask) * nimble_texture

    #     # シーンをレンダリング
    #     original_render = runner.render_original_scene()
    #     rendered_blended, rendered_albedo = runner.render_scene(hand_vertices, object_vertices)
    #     depth_image = runner.render_depth(hand_vertices, object_vertices)

    #     # スケルトンの描画
    #     skeleton_path = runner.render_skeleton(i)

    #     # 画像を保存し、パスを取得
    #     # image_paths = save_images(original_render, rendered_blended, runner.background_image, depth_image, 'test_data/', str(i))
    #     image_paths = save_images(original_render, rendered_blended, rendered_albedo, runner.background_image, depth_image, output_dir, str(i))

    #     # パスをリストに追加
    #     paths_list.append({
    #         'image': image_paths['rgb'],
    #         'seg': image_paths['seg'],
    #         'depth': image_paths['depth'],
    #         'texture': image_paths['texture'],
    #         'albedo': image_paths['albedo'],
    #         'skeleton': skeleton_path
    #     })

    # # 最後にCSVに書き込み
    # os.chdir('test_data_ad')
    # csv_file_path = 'output_paths.csv'
    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     fieldnames = ['image', 'seg', 'depth', 'texture', 'albedo', 'skeleton']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for paths in paths_list:
    #         writer.writerow(paths)

# %%
if __name__ == '__main__':
    main()

# %%
