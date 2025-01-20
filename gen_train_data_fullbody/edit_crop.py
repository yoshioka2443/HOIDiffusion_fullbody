# %%

import pandas as pd

# CSVファイルのパス（入力ファイル）
input_csv_path = "/home/datasets/fullbody_train/output_paths.csv"

# 出力ファイルのパス
output_csv_path = "/home/datasets/fullbody_train/output_paths.csv"

# CSVを読み込む
columns = [
    "image", "depth", "mask", "bodymesh", "skeleton", "seg", "texture", 
    "top", "bottom", "left", "right", "sentence"
]
df = pd.read_csv(input_csv_path, header=None, names=columns)

# 指定された値に変更
df["left"] = 420
df["top"] = 0
df["right"] = 1500
df["bottom"] = 1080

# 変更後のCSVを保存
df.to_csv(output_csv_path, index=False, header=False)

print(f"変更後のデータを {output_csv_path} に保存しました。")