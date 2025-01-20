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
import plotly.graph_objects as go

# カスタムモジュールのインポート
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)
print(os.getcwd())

from gen_test_data.image_edit import display_textures
from gen_test_data.runner import Runner

# import logging
# logging.getLogger("pyredner").setLevel(logging.CRITICAL)
os.environ["PYREDNER_VERBOSE"] = "0"

# %%
def print_device_info():
    # gpuを3番目に指定
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
    else:
        print("No GPU detected.")

# %%
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print_device_info()

    # パラメータの設定
    # output_dir = 'test_data_sample'
    output_dir = 'test_data_NIMBLE'
    sequence_name = 'GPMF12'
    frame_number = 250
    
    # 複数のオブジェクトに対する処理
    replacement_object_name_list = [
        '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', 
        '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', 
        '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser', 
        '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', 
        '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
    ]
    # パスを収集するリストを初期化
    paths_list = []

    for i, object_name in enumerate(replacement_object_name_list, start=1):
        """
        --実装メモ--
        デバイスの指定 OK
        テクスチャの最適化 OK
        replace OK
        レンダリング depth, mask, albedo, texture, seg, skeleton, rgb OK
        画像を出力 depth, mask, albedo, texture, seg, skeleton, rgb OK
        手の先を修正 OK
        """
        runner = Runner(output_dir)
        # load handmesh, obj mesh, camera parameters
        # runner.load_data(sequence_name='MC1', frame_number=250, replacement_object_name='010_potted_meat_can')
        runner.load_data(sequence_name=sequence_name, frame_number=frame_number, replacement_object_name=object_name)
        runner.make_original_scene()
        runner.render_original_scene()
        # runner.estimate_envmap()
        runner.estimate_nimble()
        runner.render_scene()
        runner.replace_object()
        runner.make_replaced_scene()
        # runner.render_scene("replaced", "estimated")
        runner.render_scene("replaced", tex="NIMBLE")
        # runner.render_scene("replaced", tex="blended")
        runner.save_images(i)

        image_paths = runner.get_paths()

        # パスをリストに追加
        paths_list.append({
            'image': image_paths['rgb'],
            'seg': image_paths['seg'],
            'depth': image_paths['depth'],
            'skeleton': image_paths['skeleton'],
            'albedo': image_paths['albedo'],
            'texture': image_paths['texture']
        })
    
    # 最後にCSVに書き込み
    os.chdir(output_dir)
    csv_file_path = 'output_paths.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'seg', 'depth', 'skeleton', 'albedo', 'texture']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paths in paths_list:
            writer.writerow(paths)
    
# %%
if __name__ == '__main__':
    main()