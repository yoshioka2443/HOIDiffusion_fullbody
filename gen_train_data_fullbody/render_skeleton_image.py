# %%
import numpy as np
import cv2
import argparse
from PIL import Image
import smplx
import trimesh
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import torch
import json
from smplx.joint_names import JOINT_NAMES

import math

# SMPL skeleton
# from skeleton_info import keypoint_info, skeleton_info 
# from SMPLX2mmpose import convert_smplx_to_mmpose

# keypoint_infoの全体確認
# print("keypoint_info:", keypoint_info)

# # itemsのリスト化
# items_list = list(keypoint_info.items())
# print("items_list:", items_list)

# # 最初のアイテムのキーと値の確認
# print("First item key:", items_list[0][0])  # キー
# print("First item value:", items_list[0][1])  # 値

# # 値の中の "name" キーを確認
# print("First item's name:", items_list[0][1]["name"])

# # SMPLXのスケルトン構造
# joint_dict = {name: i for i, name in enumerate(JOINT_NAMES)}
# print(joint_dict)

# items_list = list(keypoint_info.items())
# joint_dict2 = {i: verts_dict["name"] for i, (_, verts_dict) in enumerate(items_list)}
# print(joint_dict2)

# ゴリ押し
limbs_names = [
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("pelvis", "spine1"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("spine1", "spine2"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("spine2", "spine3"),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
    ("spine3", "neck"),
    ("neck", "left_collar"),
    ("neck", "right_collar"),
    ("neck", "head"),
    ("left_collar", "left_shoulder"),
    ("right_collar", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("head", "jaw"),
    ("head", "left_eye_smplhf"),
    ("head", "right_eye_smplhf"),
]
LIMBS = [ (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (15, 22), (15, 23), (15, 24)]

# リムのペア
# LIMBS = [
#     (0, 1), (0, 2), (0, 3),  # 中心軸
#     (2, 3), (3, 4), (4, 8),  # 右脚
#     (5, 6), (6, 7), (7, 9),  # 左脚
#     (0, 12), (12, 15),       # 上半身
#     (12, 17), (17, 18), (18, 19), (19, 20),  # 右腕
#     (12, 14), (14, 16), (16, 21), (21, 22)   # 左腕
# ]

LEFT_HAND_LIMBS = [ (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36), (20, 37), (37, 38), (38, 39)]
RIGHT_HAND_LIMBS = [ (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51), (21, 52), (52, 53), (53, 54)]
LIMBS += LEFT_HAND_LIMBS
LIMBS += RIGHT_HAND_LIMBS



def transform_world_to_camera(points_3d, R, T):
    """
    ワールド座標をカメラ座標系に変換
    :param points_3d: ワールド座標 (N, 3)
    :param R: 回転行列 (3, 3)
    :param T: 平行移動ベクトル (3,)
    :return: カメラ座標系での3Dポイント
    """
    # 平行移動を先に適用
    points_3d_translated = points_3d - T  # ワールド原点からカメラ原点への移動
    # 回転を適用
    points_camera = (R @ points_3d_translated.T).T
    return points_camera

def project_3D_to_2D(points_3d, intrinsics, cut_negativeZ=True):
    """
    3Dポイントを2Dに投影
    :param points_3d: 3D関節座標 (N, 3)
    :param intrinsics: 内部パラメータ (3, 3)
    :return: 2D座標 (N, 2)
    """
    # 投影時にZ軸が正であることを確認
    if cut_negativeZ:
        points_3d = points_3d[points_3d[:, 2] > 0]
        assert np.all(points_3d[:, 2] > 0), "全ての3D点のZ軸は正である必要があります"
    
    proj_points = intrinsics @ points_3d.T
    proj_points = proj_points[:2, :] / proj_points[2, :]  # zで割る
    return proj_points.T

def draw_skeleton(image, joints_2d=None, vertices_2d=None, object_vertices_2d=None):
    """
    スケルトンを画像に描画
    :param image: 背景画像 (numpy array)
    :param joints_2d: 2D関節座標
    """
    image = image.copy()
    
    if joints_2d is not None:
        for i, joint in enumerate(joints_2d):
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(image, (x, y), 20, (0, 255, 0), 5)
            cv2.putText(image, f"{JOINT_NAMES[i]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

        for limb in LIMBS:
            x1, y1 = int(joints_2d[limb[0], 0]), int(joints_2d[limb[0], 1])
            x2, y2 = int(joints_2d[limb[1], 0]), int(joints_2d[limb[1], 1])
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
    if vertices_2d is not None:
        for vertex in vertices_2d:
            x, y = int(vertex[0]), int(vertex[1])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    if object_vertices_2d is not None:
        for vertex in object_vertices_2d:
            x, y = int(vertex[0]), int(vertex[1])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    return image

def draw_skeleton_for_train(image, joints_2d=None):

    # 指ごとに色を設定
    # imgIn = np.copy(image)
    imgIn = np.zeros_like(image)

    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    # limbs = [[0, 1],[1, 2],[2, 3],[3,17],[0, 4],[4, 5],[5, 6],[6, 18],[0, 7],[7, 8],[8, 9],[9, 19],
    #             [0, 10],[10, 11],[11, 12],[12, 20],[0, 13],[13, 14],[14, 15],[15, 16]]

    limbs = LIMBS
    print("limbs_len=", len(limbs))
    
    gtIn = joints_2d
    # print("gtIn.shape", gtIn.shape)
    # print("joints_2d", joints_2d.shape)
    # print("gtIn.shape[0]", gtIn.shape[0])

    for joint_num in range(gtIn.shape[0]):
        # color_code_num = (joint_num // color_num)
        # joint_color = [c + 35 * (joint_num % color_num) for c in joint_color_code[color_code_num]]

        if joint_num < 24:
            color_num = 4
            color_code_num = (joint_num // color_num)
            joint_color = [c + 35 * (joint_num % color_num) for c in joint_color_code[color_code_num]]
        elif joint_num < 39:
            color_num = 3
            color_code_num = ((joint_num - 24) // color_num)
            joint_color = [c + 35 * ((joint_num - 24) % color_num) for c in joint_color_code[color_code_num]]
        elif joint_num < 54:
            color_num = 3
            color_code_num = ((joint_num - 39) // color_num)
            joint_color = [c + 35 * ((joint_num - 39) % color_num) for c in joint_color_code[color_code_num]]
        else:
            joint_color = [255, 255, 255]

        # cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
        # x, y は int にキャストしてタプルで渡す
        print("joint_num", joint_num)
        
        try:
            x = int(joints_2d[joint_num, 0])
            y = int(joints_2d[joint_num, 1])
            # cv2.circle(imgIn, center=(x, y), radius=3, color=joint_color, thickness=-1)
            cv2.circle(imgIn, center=(x, y), radius=5, color=joint_color, thickness=-1)
        except:
            print("error", joint_num)

    for limb_num in range(len(limbs)):
        print("limb_num", limb_num)
        x1 = gtIn[limbs[limb_num][0], 1]
        y1 = gtIn[limbs[limb_num][0], 0]
        x2 = gtIn[limbs[limb_num][1], 1]
        y2 = gtIn[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # if 5 < length < 150:
        #     deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        #     polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
        #                                 (int(length / 2), 3),
        #                                 int(deg),
        #                                 0, 360, 1)
        #     color_code_num = limb_num // color_num
        #     limb_color = [c + 35 * (limb_num % color_num) for c in joint_color_code[color_code_num]]
        #     cv2.fillConvexPoly(imgIn, polygon, color=limb_color)
        # else:
        #     print("length", length)
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                    (int(length / 2), 3),
                                    int(deg),
                                    0, 360, 1)
    
        # color_code_num = limb_num // color_num
        # limb_color = [c + 35 * (limb_num % color_num) for c in joint_color_code[color_code_num]]
        if limb_num < 24:
            color_num = 4
            color_code_num = (limb_num // color_num)
            limb_color = [c + 35 * (limb_num % color_num) for c in joint_color_code[color_code_num]]
        elif limb_num < 39:
            color_num = 3
            color_code_num = ((limb_num - 24) // color_num)
            limb_color = [c + 35 * ((limb_num - 24) % color_num) for c in joint_color_code[color_code_num]]
        elif limb_num < 54:
            color_num = 3
            color_code_num = ((limb_num - 39) // color_num)
            limb_color = [c + 35 * ((limb_num - 39) % color_num) for c in joint_color_code[color_code_num]]
        else:
            limb_color = [255, 255,255]

        cv2.fillConvexPoly(imgIn, polygon, color=limb_color)
    return imgIn

def make_joints(
        vertices_path = "/home/datasets/arctic/data/arctic_data/data/raw_seqs/s01/box_grab_01.smplx.npy",
        frame_idx=300, 
        smplx_model_path = "/home/datasets/arctic/data/body_models/smplx", 
        gender = 'female', 
        vtemplate_path = "/home/datasets/arctic/data/arctic_data/data/meta/subject_vtemplates/s01.obj",
        ):
    # 1. SMPLXレイヤーを初期化
    # smplxモデルのディレクトリを指定してください    

    # 2. 入力データの準備
    # body_pose: 69次元 (23関節 × 3軸の回転)
    # global_orient: 3次元 (全身の回転)
    # transl: 3次元 (全身の位置)
    # 必要に応じて`.npy`ファイルからデータを読み込みます

    data = np.load(vertices_path, allow_pickle=True)
    data = data.item()
    body_pose = data['body_pose']
    global_orient = data['global_orient']
    transl = data['transl']

    # torch.Tensorに変換
    body_pose = torch.tensor(body_pose, dtype=torch.float32)
    global_orient = torch.tensor(global_orient, dtype=torch.float32)
    transl = torch.tensor(transl, dtype=torch.float32)
    left_hand_pose = torch.tensor(data['left_hand_pose'], dtype=torch.float32)
    right_hand_pose = torch.tensor(data['right_hand_pose'], dtype=torch.float32)
    jaw_pose = torch.tensor(data['jaw_pose'], dtype=torch.float32)
    leye_pose = torch.tensor(data['leye_pose'], dtype=torch.float32)
    reye_pose = torch.tensor(data['reye_pose'], dtype=torch.float32)

    body_pose = body_pose[frame_idx].unsqueeze(0)
    global_orient = global_orient[frame_idx].unsqueeze(0)
    transl = transl[frame_idx].unsqueeze(0)
    left_hand_pose = left_hand_pose[frame_idx].unsqueeze(0)
    right_hand_pose = right_hand_pose[frame_idx].unsqueeze(0)
    jaw_pose = jaw_pose[frame_idx].unsqueeze(0)
    leye_pose = leye_pose[frame_idx].unsqueeze(0)
    reye_pose = reye_pose[frame_idx].unsqueeze(0)

    # データの形状を確認
    print("body_pose shape:", body_pose.shape) # (1, 69)
    print("global_orient shape:", global_orient.shape) # (1, 3)
    print("transl shape:", transl.shape) # (1, 3)
    print(f"{left_hand_pose.shape=}")
    print(f"{right_hand_pose.shape=}")

    # batch_size = body_pose.shape[0]  # バッチサイズ
    batch_size = 1
    print(f"{batch_size=}")
    
    v_template = trimesh.load_mesh(vtemplate_path)

    # SMPLXレイヤーを初期化
    smplx_layer = smplx.SMPLX(
        model_path=smplx_model_path, 
        gender=gender, 
        v_template=v_template.vertices,
        batch_size=batch_size, 
        use_pca=False)

    # 3. SMPLXモデルに入力して関節を取得
    smplx_output = smplx_layer(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        betas=torch.zeros(batch_size, 10),  # 10個のベータ値
        expression=torch.zeros(batch_size, 10),  # 10個の表情ベクトル
        jaw_pose=jaw_pose,  # 顎のポーズ
    )

    # 4. 関節位置を取得
    joints = smplx_output.joints  # (1, 127, 3) - 127個の関節の3D位置
    # joints = joints[:, :25]  # 25個の関節のみを取得
    joints = joints[:, :55]  # 55個の関節のみを取得
    return joints.detach().numpy()[0], smplx_output.vertices.detach().numpy()[0], smplx_layer.faces

def main(
    npy_file, 
    camera_file, 
    image_file=None, 
    output_file=None, 
    object_file=None, 
    object_mesh_file=None, 
    misc_file=None,
    subject='s01',
    frame_idx=299, 
    show_plotly=False, 
    camera_idx=0):
    """
    メイン処理
    :param npy_file: 3D座標が保存されたnpyファイル
    :param camera_file: カメラパラメータが保存されたnpyファイル
    :param image_file: 背景画像ファイル
    :param output_file: 出力画像ファイル
    """
    print(npy_file, camera_file, image_file, output_file, frame_idx)
    
    if misc_file is not None:
        misc_data = json.load(open(misc_file))
        print(misc_data)
        
        cams_Rt = np.array(misc_data[subject]['world2cam'])
        cams_K = np.array(misc_data[subject]['intris_mat'])
        print(f"{cams_Rt.shape=}")
        print(f"{cams_K.shape=}")
        
        gender = misc_data[subject]['gender']
    else:
        gender = 'female'
    
    # 3D関節を読み込み
    # joints_3d = np.load(npy_file)  # shape: (NUM_JOINTS, 3)

    joints_3d, smplx_vertices, smplx_faces = make_joints(npy_file, frame_idx=frame_idx, gender=gender)
    print(f"{joints_3d.shape=}")
    print(f"{smplx_vertices.shape=}")

    # カメラパラメータを読み込み
    camera_data = np.load(camera_file, allow_pickle=True).item()
    print(camera_data.keys())

    if camera_idx == 0:  # egocentric camera
        # 内部パラメータ（intrinsics）
        intrinsics = np.array(camera_data['intrinsics'])  # shape: (3, 3)

        # 外部パラメータ（R: 回転行列, T: 平行移動ベクトル）
        R = camera_data['R_k_cam_np'][frame_idx]  # shape: (3, 3)
        T = camera_data['T_k_cam_np'][frame_idx]  # shape: (3,)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:] = T
    else:
        cam1_Rt = cams_Rt[camera_idx-1]
        cam1_K = cams_K[camera_idx-1]
        
        w2c = np.eye(4)
        w2c[:3, :3] = cam1_Rt[:3, :3] 
        w2c[:3, 3:] = cam1_Rt[:3, 3:]
        intrinsics = cam1_K    
    
    c2w = np.linalg.inv(w2c)
    
    
    ego_markers = np.array(camera_data['ego_markers.ref'])
    
    
    # objectを読み込み
    object_data = np.load(object_file, allow_pickle=True)
    object_mesh = trimesh.load_mesh(object_mesh_file)
    object_vertices = object_mesh.vertices
    
    object_rotvec = object_data[frame_idx][1:4]
    object_trans = object_data[frame_idx][4:7]
    object_rmat = Rotation.from_rotvec(object_rotvec).as_matrix()
    
    object_vertices = (object_vertices @ object_rmat.T + object_trans) * 0.001
    object_vertices_camera = object_vertices @ w2c[:3, :3].T + w2c[:3, 3:].T
    object_vertices_2d = project_3D_to_2D(object_vertices_camera, intrinsics, cut_negativeZ=False)

    # ワールド座標をカメラ座標系に変換
    # joints_camera = transform_world_to_camera(joints_3d, R, T)
    joints_camera = joints_3d @ w2c[:3, :3].T + w2c[:3, 3:].T
    
    # 2D座標に投影
    joints_2d = project_3D_to_2D(joints_camera, intrinsics, cut_negativeZ=False)

    # smplxメッシュを2D座標に変換    
    vertices_camera = smplx_vertices @ w2c[:3, :3].T + w2c[:3, 3:].T
    vertices_2d = project_3D_to_2D(vertices_camera, intrinsics, cut_negativeZ=False)

    # # 仮のカメラ行列（必要に応じて修正）
    # camera_matrix = np.array([
    #     [500, 0, 256],  # fx, 0, cx
    #     [0, 500, 256],  # 0, fy, cy
    #     [0, 0, 1]       # 0, 0, 1
    # ])

    # 背景画像を読み込み
    if image_file is None:
        image = np.ones((int(intrinsics[1,2]*2), int(intrinsics[0,2]*2), 3), dtype=np.uint8) * 255
    else:
        image = np.array(Image.open(image_file))
        print(f"{image.shape=}")
    
    dist_coeffs = np.array(camera_data['dist8'])
    # print(f"{intrinsics=}")
    # image = cv2.undistort(image, intrinsics, dist_coeffs)  

    # スケルトンを描画
    # joints_2d = None
    # object_vertices_2d = None
    # output_image = draw_skeleton(image, joints_2d, vertices_2d, object_vertices_2d)
    output_image = draw_skeleton_for_train(image, joints_2d)
    
    print(f'{output_image.shape=}')

    # 出力
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(image)
    axs[1].imshow(output_image)
    plt.show()
        
    if show_plotly:
        fig = go.Figure()
        # set orthographic camera
        fig.update_layout(scene=dict(camera=dict(projection=dict(type="orthographic"))))    
        
        # smplx jointsを描画
        fig.add_trace(
            go.Scatter3d(
                x=joints_3d[:, 0], y=joints_3d[:, 1], z=joints_3d[:, 2], 
                mode='markers', marker=dict(size=5))
                # mode='markers+text', marker=dict(size=5), text=JOINT_NAMES)
            )
        
        # smplxメッシュを描画
        fig.add_trace(
            go.Mesh3d(
                x=smplx_vertices[:, 0], y=smplx_vertices[:, 1], z=smplx_vertices[:, 2],
                i=smplx_faces[:, 0], j=smplx_faces[:, 1], k=smplx_faces[:, 2],
                color='lightpink', opacity=0.5
            )
            )
        
        # カメラワイヤーフレームを描画
        camera_wireframe = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [0,0,0],
            [0,0,2],
        ]) * 0.2
        camera_wireframe = camera_wireframe @ c2w[:3,:3].T + c2w[:3,3:].T

        fig.add_trace(
            go.Scatter3d(
                x=camera_wireframe[:, 0],
                y=camera_wireframe[:, 1],
                z=camera_wireframe[:, 2],
                mode='lines',
            )
            )
        
        # objectを描画
        fig.add_trace(
            go.Mesh3d(
                x=object_vertices[:, 0],
                y=object_vertices[:, 1],
                z=object_vertices[:, 2],
                i=object_mesh.faces[:, 0],
                j=object_mesh.faces[:, 1],
                k=object_mesh.faces[:, 2],
            ))
        fig.show()
    
    cv2.imwrite(output_file, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"スケルトン画像を保存しました: {output_file}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="SMPLXスケルトンの可視化")
    # parser.add_argument("--npy_file", type=str, required=True, help="SMPLX 3D関節座標のnpyファイルパス")
    # parser.add_argument("--camera_file", type=str, required=True, help="カメラパラメータのnpyファイルパス")
    # parser.add_argument("--image_file", type=str, required=True, help="背景画像のパス")
    # parser.add_argument("--output_file", type=str, required=True, help="出力画像のパス")

    # args = parser.parse_args()
    # main(args.npy_file, args.camera_file, args.image_file, args.output_file)
    from pathlib import Path
    
    # data_dir = Path("/home/datasets/arctic/data/arctic_data/data")
    # data_dir = Path("./data/arctic_data/data")
    data_dir = Path("/home/datasets/arctic/data/arctic_data/data")

    subject = "s01"
    action = "box_grab_01"
    object_name = "box"
    frame_idx = 299
    camera_idx = 0
    
    npy_file = data_dir / f"raw_seqs/{subject}/{action}.smplx.npy"
    camera_file = data_dir / f"raw_seqs/{subject}/{action}.egocam.dist.npy"
    object_mesh_file = data_dir / f"meta/object_vtemplates/{object_name}/mesh.obj"
    # object_mesh_file = data_dir / "meta/subject_vtemplates/s01.obj"
    object_file = data_dir / f"raw_seqs/{subject}/{action}.object.npy"
    misc_file = data_dir / "meta/misc.json"
    # image_file = data_dir / "cropped_images/s01/box_grab_01/0/00300.jpg"
    image_file = f"/home/datasets/arctic/data/arctic_data/data/images/{subject}/{action}/{camera_idx}/{frame_idx+1:05d}.jpg"
    output_file = "skeleton_output.png"
    # npy_file = "box_grab_01.smplx.npy"
    # camera_file = "0/box_grab_01.egocam.dist.npy"
    # object_file = '0/box_grab_01.object.npy'
    # object_mesh_file = 'mesh.obj'
    # # misc_file = 'misc.json'
    # image_file = "0/00300.jpg"
    # output_file = "skeleton_output.png"

    main(
        npy_file=npy_file, 
        camera_file=camera_file, 
        image_file=image_file, 
        output_file=output_file, 
        misc_file=misc_file, 
        object_file=object_file, 
        object_mesh_file=object_mesh_file, 
        frame_idx=frame_idx, 
        subject=subject,
        show_plotly=True, 
        camera_idx=camera_idx)
    
    # allocentric camera
    # image_file = "1/00300.jpg"
    # image_file = data_dir / "cropped_images/s01/box_grab_01/1/00300.jpg"
    camera_idx = 1    
    image_file = f"/home/datasets/arctic/data/arctic_data/data/images/{subject}/{action}/{camera_idx}/{frame_idx+1:05d}.jpg"
    output_file = "skeleton_output_allocentric.png"
    main(
        npy_file=npy_file, 
        camera_file=camera_file, 
        image_file=image_file, 
        output_file=output_file, 
        misc_file=misc_file, 
        object_file=object_file, 
        object_mesh_file=object_mesh_file, 
        frame_idx=frame_idx, 
        subject=subject,
        show_plotly=True, 
        camera_idx=camera_idx)
    


# %%
camera_idx = 2    
image_file = f"/home/datasets/arctic/data/arctic_data/data/images/{subject}/{action}/{camera_idx}/{frame_idx+1:05d}.jpg"
output_file = f"skeleton_output_allocentric_{camera_idx}.png"
main(
    npy_file=npy_file, 
    camera_file=camera_file, 
    image_file=image_file, 
    output_file=output_file, 
    misc_file=misc_file, 
    object_file=object_file, 
    object_mesh_file=object_mesh_file, 
    frame_idx=frame_idx, 
    subject=subject,
    show_plotly=True, 
    camera_idx=camera_idx)
# %%