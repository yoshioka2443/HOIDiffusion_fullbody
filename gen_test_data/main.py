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
        torch.cuda.set_device(2)
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
    output_dir='sample1'
    
    runner = Runner(output_dir)
    # load handmesh, obj mesh, camera parameters
    runner.load_data(sequence_name='MC1', frame_number=250, replacement_object_name='010_potted_meat_can')
    # runner.load_data(sequence_name='GPMF12', frame_number=250, replacement_object_name='010_potted_meat_can')
    runner.make_original_scene()
    runner.render_original_scene()
    runner.estimate_envmap()
    runner.render_scene()
    runner.save_images()

    """
    デバイスの指定
    テクスチャの最適化
    replace
    レンダリング depth, mask, albedo, texture, seg, skeleton, rgb
    画像を出力 depth, mask, albedo, texture, seg, skeleton, rgb
    手の先を修正

    """
# %%
if __name__ == '__main__':
    main()