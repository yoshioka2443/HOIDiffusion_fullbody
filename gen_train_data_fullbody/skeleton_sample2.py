# %%
import matplotlib.pyplot as plt
import numpy as np

# 図の中心座標とスケルトンデータ
joints = {
    "torso": [(0, 0), (0, 1)],
    "left_arm": [(0, 1), (-1, 2), (-1.5, 2.5)],
    "right_arm": [(0, 1), (1, 2), (1.5, 2.5)],
    "left_leg": [(0, 0), (-0.5, -1), (-0.75, -1.5)],
    "right_leg": [(0, 0), (0.5, -1), (0.75, -1.5)],
}

# 色
colors = {
    "torso": "#FF0000",
    "left_arm": "#0000FF",
    "right_arm": "#00FF00",
    "left_leg": "#FFA500",
    "right_leg": "#800080",
}

# 描画
plt.figure(figsize=(6, 6))

# スケルトンの各パーツを描画
for part, coords in joints.items():
    x, y = zip(*coords)
    plt.plot(x, y, marker='o', color=colors[part], label=part)

# 放射状デザインのデコレーション（例として正弦波を追加）
angles = np.linspace(0, 2 * np.pi, 100)
for i in range(5):
    r = 1 + 0.1 * i
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    plt.plot(x, y, color='lightgray', linestyle='--', alpha=0.5)

# 設定
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()
