# %%
from smplx import SMPLX
import trimesh
import pickle

# SMPLXモデルのロード
model_path = "/home/datasets/arctic/data/body_models/smplx/SMPLX_NEUTRAL.pkl"
# modle_path = "/home/datasets/arctic/data/body_models/smplx/SMPLX_NEUTRAL.npz"


with open(model_path, 'rb') as f:
    model_data = pickle.load(f, encoding='latin1')  # 必要に応じてエンコーディングを指定
print(model_data.keys()) 


regressor = model_data['J_regressor']
# model = SMPLX(model_path=model_path, batch_size=1)

# # リグレッサの取得
# # J_regressor: scipy.sparse.csr_matrix (ジョイントリグレッサの行列形式)
# regressor = model.J_regressor.toarray() 
print(regressor.shape)  # (24, 10475)


verts_file_path = "/home/datasets/GOAL/results/apple_grasp/0000_sbj_refine.ply"
bodymesh = trimesh.load(verts_file_path)

print(bodymesh.vertices.shape)  # (10475, 3)

joints = regressor @ bodymesh.vertices

print(joints.shape)  # (24, 3)
# %%
