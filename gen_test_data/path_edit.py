# %%
import pandas as pd

# CSVファイルのパス（変更してください）
input_csv = "/home/datasets/test_data_NIMBLE/output_paths.csv"  # 入力するCSVファイルのパス
output_csv = "output_paths.csv"  # 出力するCSVファイルのパス

# root_dir（追加したいディレクトリパス）
root_dir = "/home/datasets"  # 必要に応じて変更

# CSVファイルを読み込み
df = pd.read_csv(input_csv)

# 各列のパスにroot_dirを追加
for column in df.columns:
    df[column] = df[column].apply(lambda x: f"{root_dir}/{x}")

# 変更後のCSVを保存
df.to_csv(output_csv, index=False)
print(f"Updated CSV saved to: {output_csv}")
