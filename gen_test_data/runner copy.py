import os
import sys
import pickle
import math
import numpy as np

import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pyredner
from jupyterplot import ProgressPlot

from grabnet.tools.meshviewer import Mesh as M
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler, aa2rotmat, makepath, to_cpu
from grabnet.tools.cfg_parser import Config
from grabnet.tools.train_tools import point2point_signed
from grabnet.tests.tester import Tester
from NIMBLE.manolayer import ManoLayer
import mano
from bps_torch.bps import bps_torch

from tools.run_grabnet import run_grabnet
from tools.plotly import PlotlyFigure
from tools.load_mesh import load_obj_verts_, load_ho_meta
from tools.transform import apply_transform_to_mesh, apply_trans
from tools.texture import calc_vertex_normals, calc_face_normals

from gen_test_data.relighting import deringing
from gen_test_data.transform import apply_transform, to_homogeneous_matrix
from gen_test_data.image_edit import resize_and_pad_image

import cv2

os.environ["PYREDNER_VERBOSE"] = "0"

def save_image(original_image, output_dir, filename="original.png"):
    """画像を保存します。"""
    plt.show()
    plt.imsave(os.path.join(output_dir, filename), original_image)
    print(f"Saved original image to {os.path.join(output_dir, filename)}")
    plt.close()

class Runner:
    def __init__(self):
        """GrabNetと関連モデルを初期化します。"""
        self.initialize_grabnet()

    def initialize_grabnet(self):
        """GrabNetと関連モデルを初期化します。"""
        config_path = 'grabnet/configs/grabnet_cfg.yaml'
        mano_right_hand_model_path = 'mano_data/mano_v1_2/models/mano/MANO_RIGHT.pkl'
        config = dict(
            work_dir='grabnet/logs',
            best_cnet='grabnet/models/coarsenet.pt',
            best_rnet='grabnet/models/refinenet.pt',
            bps_dir='grabnet/configs/bps.npz',
            rhm_path=mano_right_hand_model_path
        )
        grabnet_config = Config(default_cfg_path=config_path, **config)

        self.grabnet = Tester(cfg=grabnet_config)
        self.grabnet.coarse_net.eval()
        self.grabnet.refine_net.eval()
        self.coarse_net = self.grabnet.coarse_net
        self.refine_net = self.grabnet.refine_net

        self.right_hand_model = mano.load(
            model_path=self.grabnet.cfg.rhm_path,
            model_type='mano',
            num_pca_comps=45,
            batch_size=1,
            flat_hand_mean=True
        )

        self.grabnet.refine_net.rhm_train = self.right_hand_model
        self.bps = bps_torch(custom_basis=self.grabnet.bps)

    def load_data(self, sequence_name='GPMF12', frame_number=250, replacement_object_name='011_banana'):
        """データを読み込み、必要なパラメータを初期化します。"""
        self.load_background_image(sequence_name, frame_number)
        self.load_object_data(sequence_name, frame_number, replacement_object_name)
        self.setup_camera()
        print("Data loaded.")
        # self.estimate_envmap()

    def load_background_image(self, sequence_name, frame_number):
        """背景画像を読み込みます。"""
        image_path = f'data/HO3D_v3/train/{sequence_name}/rgb/{frame_number:04d}.jpg'
        self.background_image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        # self.background_envmap = pyredner.EnvironmentMap(torch.tensor(self.background_image).to(pyredner.get_device()))

    def load_object_data(self, sequence_name, frame_number, replacement_object_name):
        """オブジェクトと手のデータを読み込みます。"""
        meta_path = f'../dataset/HO3D_v3/train/{sequence_name}/meta/{frame_number:04d}.pkl'
        with open(meta_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.original_object_path = f"../dataset/models/{data['objName']}/textured_simple.obj"
        self.replacement_object_path = f"../dataset/models/{replacement_object_name}/textured_simple.obj"

        # オブジェクトと手のパラメータ
        self.object_rotation_np = R.from_rotvec(data['objRot'].T[0]).as_matrix().astype(np.float32)
        self.object_translation_np = data['objTrans'].astype(np.float32)
        self.hand_translation_np = data['handTrans'].astype(np.float32)
        self.global_orient = data['handPose'][:3]
        self.hand_rotation_np = R.from_rotvec(self.global_orient).as_matrix().astype(np.float32)
        self.hand_pose_np = data['handPose'][3:]

        # テンソルに変換
        self.object_rotation = torch.tensor(self.object_rotation_np)
        self.object_translation = torch.tensor(self.object_translation_np)
        self.hand_rotation = torch.tensor(self.hand_rotation_np)
        self.hand_translation = torch.tensor(self.hand_translation_np)
        self.hand_pose = torch.tensor(self.hand_pose_np).unsqueeze(0)

        # オブジェクトの読み込み
        self.original_object = pyredner.load_obj(self.original_object_path, return_objects=True)[0]
        self.original_object_vertices = self.original_object.vertices.type(torch.float32).cpu() @ self.object_rotation.T + self.object_translation
        self.replacement_object = pyredner.load_obj(self.replacement_object_path, return_objects=True)[0]

        # 手のモデルをロード
        self.annotation = load_ho_meta(meta_path)
        self.mano_layer = ManoLayer()
        self.mano_layer.load_textures()
        self.mano_hand = self.mano_layer(self.annotation)
    
    def setup_camera(self):
        """カメラのパラメータを設定します。"""
        self.resolution = self.background_image.shape[:2]
        world_to_cam = torch.eye(4)
        rotation_matrix = torch.diag(torch.tensor([-1., 1., -1.]))
        world_to_cam[:3, :3] = rotation_matrix
        self.K = torch.tensor(self.annotation['camMat'], dtype=torch.float32)
        fx, fy = self.K.diagonal()[:2]
        px, py = self.K[:2, 2]

        intrinsic_mat = torch.tensor([
            [fx / self.resolution[1] * 2, 0.0, px / self.resolution[1] - 0.5],
            [0.0, fy / self.resolution[1] * 2, py / self.resolution[0] - 0.5],
            [0.0, 0.0, 1.0]
        ])

        self.camera = pyredner.Camera(
            intrinsic_mat=intrinsic_mat,
            position=torch.tensor([0, 0, 0.], dtype=torch.float32),
            look_at=torch.tensor([0, 0, -1.], dtype=torch.float32),
            up=torch.tensor([0, 1., 0], dtype=torch.float32),
            resolution=self.resolution,
        )

    def make_original_scene(self):
        uvs = torch.stack([self.mano_layer.uv[..., 0], 1 - self.mano_layer.uv[..., 1]], -1)

        # 元の手のメッシュ
        self.original_hand_vertices = self.mano_hand.vertices[0].type(torch.float32).cpu().detach()
        vertex_normals = calc_vertex_normals(self.original_hand_vertices, self.mano_layer.faces)

        self.original_hand = pyredner.Object(
            vertices=self.original_hand_vertices,
            indices=self.mano_layer.faces.to(torch.int32),
            uvs=torch.tensor(uvs, dtype=torch.float32),
            uv_indices=torch.tensor(self.mano_layer.face_uvs, dtype=torch.int32),
            normals=torch.tensor(vertex_normals, dtype=torch.float32),
            normal_indices=self.mano_layer.faces.to(torch.int32),
            material=pyredner.Material(
                diffuse_reflectance=self.mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
                specular_reflectance=self.mano_layer.tex_spec_mean.to(pyredner.get_device())
            ),
            material_id = 0
        )

        self.original_object = pyredner.Object(
            vertices=self.original_object_vertices.type(torch.float32),
            indices=self.original_object.indices.type(torch.int32),
            uvs=self.original_object.uvs,
            uv_indices=self.original_object.uv_indices,
            material=self.original_object.material,
            material_id = 1
        )

        self.scene = pyredner.Scene(
            camera=self.camera,
            objects=[self.original_hand, self.original_object]
        )
        print("Original scene created.")

    def render_original_scene(self):
        """置き換え前のシーンをレンダリングします。"""
        light = pyredner.AmbientLight(intensity=torch.tensor([1., 1., 1.]))
        self.original_scene = pyredner.render_deferred(self.scene, lights=[light], alpha=True)
        print("Original scene rendered.")
        return self.original_scene
    
    def save_images(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_image = torch.pow(self.original_scene, 1.0 / 2.2).cpu().detach().numpy()
        save_image(original_image, output_dir, filename="original.png")
        save_image(self.background_image, output_dir, filename="background.png")
        save_image(self.replaced_scene, output_dir, filename="replaced.png")
        save_image(self.replaced_albedo, output_dir, filename="albedo.png")
        print(f"Images saved to {output_dir}")


    def estimate_envmap(self):
        print(type(self.original_object))
        materials = [
            pyredner.Material(
                diffuse_reflectance = self.mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
                specular_reflectance = self.mano_layer.tex_spec_mean.to(pyredner.get_device()),
                normal_map = self.mano_layer.tex_normal_mean.to(pyredner.get_device()),
            ),
            self.original_object.material,
        ]
        self.coeffs_sh = torch.zeros((3, 16), device=pyredner.get_device())
        self.coeffs_sh[:, 0] += 0.5
        self.coeffs_sh.requires_grad = True
        self.deringed_coeffs_sh = deringing(self.coeffs_sh, 6.0)
        # self.envmap = pyredner.SH_reconstruct(self.deringed_coeffs_sh, self.resolution)

        self.coeffs_tex = torch.zeros(10, device=pyredner.get_device(), requires_grad=True)
        ##
        optimizer = torch.optim.Adam([
            self.coeffs_sh, 
            self.coeffs_tex
            ], 
            lr=3e-2)

        scene_args = pyredner.RenderFunction.serialize_scene(
            scene = self.scene,
            num_samples = 512,
            max_bounces = 1,
            sampler_type=pyredner.sampler_type.independent,
            # sampler_type = pyredner.sampler_type.sobol,
            use_primary_edge_sampling=True,
            use_secondary_edge_sampling=True,
            channels = [
                pyredner.channels.radiance, 
                # pyredner.channels.alpha, 
                pyredner.channels.geometry_normal,
                pyredner.channels.shading_normal,
                # pyredner.channels.diffuse_reflectance,
                ]
            )
        render = pyredner.RenderFunction.apply
        # Render the scene.
        render_albedo = pyredner.render_albedo(self.scene, alpha=True)
        mask = render_albedo[..., -1]

        
        target = torch.pow(torch.tensor(self.background_image, device=pyredner.get_device()), 2.2)

        lambda_reg_tex = 1e2
        lambda_reg_sh = 1e2

        pp = ProgressPlot(line_names=['loss', 'loss_mse', 'reg_tex', 'reg_sh'], plot_names=['loss'])
        # pp_coeffs = ProgressPlot(line_names=[f'coeff{i}' for i in range(coeffs_tex.shape[0])])
        for t in range(100):
            # print('iteration:', t)
            optimizer.zero_grad()
            # Repeat the envmap generation & material for the gradients
            deringed_coeffs_sh = deringing(self.coeffs_sh, 6.0)
            envmap = pyredner.SH_reconstruct(deringed_coeffs_sh, self.resolution)
            envmap = pyredner.EnvironmentMap(envmap)
            diffuse_reflectance = torch.sum(self.coeffs_tex * self.mano_layer.tex_diffuse_basis.to(pyredner.get_device()), dim=-1) + self.mano_layer.tex_diffuse_mean.to(pyredner.get_device())
            specular_reflectance = torch.sum(self.coeffs_tex * self.mano_layer.tex_spec_basis.to(pyredner.get_device()), dim=-1) + self.mano_layer.tex_spec_mean.to(pyredner.get_device())
            normal_map = torch.sum(self.coeffs_tex * self.mano_layer.tex_normal_basis.to(pyredner.get_device()), dim=-1) + self.mano_layer.tex_normal_mean.to(pyredner.get_device())
            materials[0] = pyredner.Material(
                diffuse_reflectance = diffuse_reflectance,
                specular_reflectance = specular_reflectance,
                # normal_map = normal_map,
            )
            # mano_shape = pyredner.Shape(
            #     vertices = mano_vertices, 
            #     indices = mano_indices, 
            #     uvs = torch.tensor(uvs, dtype=torch.float32),
            #     uv_indices = torch.tensor(mano_layer.face_uvs, dtype=torch.int32),
            #     normals = vertex_normals, 
            #     material_id = 0
            # )

            # obj_shape = pyredner.Shape(
            #     vertices=apply_transform_to_mesh(objects[0].vertices, anno),
            #     indices=objects[0].indices, 
            #     uvs=objects[0].uvs,
            #     uv_indices=objects[0].uv_indices,
            #     material_id=1,
            # )
            # shapes = [mano_shape, obj_shape] 
            scene = pyredner.Scene(camera = self.camera,
                                shapes = [self.original_hand.shapes[0], self.original_object.shapes[0]],
                                materials = materials,
                                envmap = envmap)
            scene_args = pyredner.RenderFunction.serialize_scene(
                            scene = scene,
                            num_samples = 4,
                            max_bounces = 1, 
                            )
            img = render(t+1, *scene_args)
            # loss_mse = torch.pow((img - target) * mask_mano.unsqueeze(-1), 2).sum()
            loss_mse = torch.pow((img - target) * mask.unsqueeze(-1), 2).sum()
            reg_tex = torch.pow(self.coeffs_tex, 2).sum() * lambda_reg_tex
            reg_sh = torch.pow(self.coeffs_sh[:, 1:], 2).sum() * lambda_reg_sh
            loss = loss_mse + reg_tex + reg_sh
            # print(f"loss_total: {loss.item():.04f}, loss_mse: {loss_mse.item():.04f}, reg_tex: {reg_tex.item():.04f}")

            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            
            pp.update({
                'loss': {
                    'loss': loss.item(),
                    'loss_mse': loss_mse.item(),
                    'reg_tex': reg_tex.item(),
                    'reg_sh': reg_sh.item(),
                },
            })

            if t > 0 and t % (10 ** int(math.log10(t))) == 0:
                pyredner.imwrite(img.cpu(), f'{save_dir}/iter_{t}.png')

        pp.finalize()

        envmap = pyredner.SH_reconstruct(self.deringed_coeffs_sh, self.resolution)
        self.envmap = pyredner.EnvironmentMap(envmap)
        print("Envmap estimated.")


    def replace_object(self):
        """オブジェクトを置き換え、手とオブジェクトの頂点を取得します。"""
        translation = (self.hand_translation - self.object_translation).unsqueeze(0)

        # 元の手のパラメータ
        original_hand_params = dict(
            transl=self.hand_translation.unsqueeze(0).to(self.grabnet.device),
            grobal_orient=self.hand_rotation.to(self.grabnet.device),
            hand_pose=self.hand_pose.to(self.grabnet.device)
        )

        # GrabNetで手を生成
        _, verts_rh_gen_cnet, verts_rh_gen_rnet, rh_gen_rnet = run_grabnet(
            self.replacement_object_path,
            self.grabnet,
            self.bps,
            self.right_hand_model.to(self.grabnet.device),
            self.coarse_net,
            self.refine_net,
            rotmat=self.object_rotation_np,
            trans_rhand=translation.to(self.grabnet.device),
            global_orent_rhand_rotmat=self.hand_rotation.to(self.grabnet.device)
        )

        # 元の手の位置
        self.hov3_hand = self.right_hand_model(**original_hand_params)
        hov3_wrist_position = self.hov3_hand.joints[:, :1, :][0].detach().cpu()
        self.gen_hand = rh_gen_rnet
        gen_wrist_position = self.gen_hand.joints[:, :1, :][0].cpu() + self.object_translation
        gen_hand_vertices = verts_rh_gen_rnet[0].cpu()

        # 生成した手の頂点を取得
        hand_grabnet_gen = gen_hand_vertices + self.object_translation
        gen_rotation = aa2rotmat(self.gen_hand.global_orient).view(-1, 3, 3)[0].cpu()

        # オブジェクトの頂点を取得
        obj_vertices = self.replacement_object.vertices.cpu().to(torch.float32)
        obj_vertices = apply_transform(obj_vertices, to_homogeneous_matrix(R=self.object_rotation, T=self.object_translation))

        # 変換行列を作成
        transform_matrix = to_homogeneous_matrix(T=hov3_wrist_position) @ \
                           to_homogeneous_matrix(R=self.hand_rotation) @ \
                           to_homogeneous_matrix(R=gen_rotation.T) @ \
                           to_homogeneous_matrix(T=-gen_wrist_position)

        # 手とオブジェクトの頂点に変換を適用
        self.mano_vertices = apply_transform(hand_grabnet_gen, transform_matrix)
        self.object_vertices = apply_transform(obj_vertices, transform_matrix)

        self.gen_hand_joints_trans = apply_transform(self.gen_hand.joints[:, :, :][0].cpu() + self.object_translation, transform_matrix)

        return self.mano_vertices, self.object_vertices

    def render_scene(self, hand_vertices, object_vertices):
        """シーンをレンダリングします。"""
        light = pyredner.AmbientLight(intensity=torch.tensor([1., 1., 1.]))
        uvs = torch.stack([self.mano_layer.uv[..., 0], 1 - self.mano_layer.uv[..., 1]], -1)
        vertex_normals = calc_vertex_normals(hand_vertices, self.mano_layer.faces)

        hand = pyredner.Object(
            vertices=hand_vertices,
            indices=self.mano_layer.faces.to(torch.int32),
            uvs=torch.tensor(uvs, dtype=torch.float32),
            uv_indices=torch.tensor(self.mano_layer.face_uvs, dtype=torch.int32),
            normals=torch.tensor(vertex_normals, dtype=torch.float32),
            normal_indices=self.mano_layer.faces.to(torch.int32),
            material=pyredner.Material(
                diffuse_reflectance=self.mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
                specular_reflectance=self.mano_layer.tex_spec_mean.to(pyredner.get_device())
            )
        )

        replacement_object = pyredner.Object(
            vertices=object_vertices,
            indices=self.replacement_object.indices.type(torch.int32),
            uvs=self.replacement_object.uvs,
            uv_indices=self.replacement_object.uv_indices,
            material=self.replacement_object.material
        )

        # scene = pyredner.Scene(
        #     camera=self.camera,
        #     objects=[hand, replacement_object]
        # )
        # return pyredner.render_deferred(scene, lights=[light], alpha=True), pyredner.render_albedo(scene)
        scene = pyredner.Scene(
            camera=self.camera,
            objects=[hand, replacement_object],
            envmap=self.envmap
        )

        self.replaced_scene = pyredner.render_deferred(scene, alpha=True)
        self.replaced_albedo = pyredner.render_albedo(scene)
        return self.replaced_scene, self.replaced_albedo
    
    def render_depth(self, hand_vertices, object_vertices):
        """シーンのデプス（深度）をレンダリングし、0から1の範囲に正規化してグレースケール画像として返します。"""
        uvs = torch.stack([self.mano_layer.uv[..., 0], 1 - self.mano_layer.uv[..., 1]], -1)
        vertex_normals = calc_vertex_normals(hand_vertices, self.mano_layer.faces)

        hand = pyredner.Object(
            vertices=hand_vertices,
            indices=self.mano_layer.faces.to(torch.int32),
            uvs=torch.tensor(uvs, dtype=torch.float32),
            uv_indices=torch.tensor(self.mano_layer.face_uvs, dtype=torch.int32),
            normals=torch.tensor(vertex_normals, dtype=torch.float32),
            normal_indices=self.mano_layer.faces.to(torch.int32),
            material=pyredner.Material(
                diffuse_reflectance=self.mano_layer.tex_diffuse_mean.to(pyredner.get_device()),
                specular_reflectance=self.mano_layer.tex_spec_mean.to(pyredner.get_device())
            )
        )
        replacement_object = pyredner.Object(
            vertices=object_vertices,
            indices=self.replacement_object.indices.type(torch.int32),
            uvs=self.replacement_object.uvs,
            uv_indices=self.replacement_object.uv_indices,
            material=self.replacement_object.material
        )
        scene = pyredner.Scene(
            camera=self.camera,
            objects=[hand, replacement_object]
        )

        g_buffer = pyredner.render_g_buffer(
            scene=scene,
            channels=[pyredner.channels.depth]
        )
        depth_map = g_buffer[..., 0]

        # デプス値を0から1に正規化
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)

        return normalized_depth



    def project_texture_to_uv(self, texture=None):
        """テクスチャをUV空間に投影します。"""
        if texture is None:
            texture = self.background_image

        world_to_cam = torch.eye(4)
        cv_to_gl = torch.diag(torch.tensor([1., -1., -1., 1]))
        world_to_cam = world_to_cam @ cv_to_gl

        vertices_cam = self.mano_hand.vertices[0] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3].T
        vertices_ndc = vertices_cam @ self.K.T
        vertices_screen = vertices_ndc[..., :-1] / vertices_ndc[..., -1:]

        vertices_uv = vertices_screen / torch.tensor([self.resolution[1], self.resolution[0]], dtype=torch.float32)

        uvs3d = (torch.stack([
            self.mano_layer.uv[..., 0],
            self.mano_layer.uv[..., 1],
            torch.zeros_like(self.mano_layer.uv[..., 0])], -1) * 2 - 1).to(pyredner.get_device()).type(torch.float32)

        mano_uv_renderer = pyredner.Object(
            vertices=uvs3d,
            indices=torch.tensor(self.mano_layer.face_uvs, dtype=torch.int32),
            uvs=vertices_uv,
            uv_indices=self.mano_layer.faces.to(torch.int32),
            material=pyredner.Material(
                diffuse_reflectance=torch.tensor(texture).type(torch.float32).to(pyredner.get_device()),
            )
        )

        camera_uv = pyredner.Camera(
            camera_type=pyredner.camera_type.orthographic,
            position=torch.tensor([0, 0, 1.], dtype=torch.float32),
            look_at=torch.tensor([0, 0, 0.], dtype=torch.float32),
            up=torch.tensor([0, 1., 0], dtype=torch.float32),
            resolution=self.mano_layer.tex_diffuse_mean.shape[:2],
        )

        return pyredner.render_albedo(pyredner.Scene(camera=camera_uv, objects=[mano_uv_renderer]))

    def compute_front_mask(self):
        """手前向きの頂点のマスクを計算します。"""
        world_to_cam = torch.eye(4)
        cv_to_gl = torch.diag(torch.tensor([1., -1., -1., 1]))
        world_to_cam = world_to_cam @ cv_to_gl

        vertices_cam = self.mano_hand.vertices[0] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3].T
        vertices_ndc = vertices_cam @ self.K.T
        vertices_screen = vertices_ndc[..., :-1] / vertices_ndc[..., -1:]
        vertices_uv = vertices_screen / torch.tensor([self.resolution[1], self.resolution[0]], dtype=torch.float32)

        vertices_normal_world = pyredner.compute_vertex_normal(self.mano_hand.vertices[0], self.mano_layer.faces)
        vertices_normal_camera = vertices_normal_world @ torch.tensor(world_to_cam)[:3, :3].T
        vertices_front = vertices_normal_camera @ torch.tensor([0, 0, -1.], dtype=torch.float32)
        vertices_front = vertices_front.unsqueeze(-1).expand(-1, 3)

        uvs3d = torch.stack([
            self.mano_layer.uv[..., 0],
            self.mano_layer.uv[..., 1],
            torch.zeros_like(self.mano_layer.uv[..., 0])], -1) * 2 - 1

        vertices_uvshape = torch.tensor(uvs3d, dtype=torch.float32)
        for ft, fv in zip(self.mano_layer.face_uvs, self.mano_layer.faces):
            vertices_uvshape[ft] = vertices_front[fv]

        mano_uv_front = pyredner.Object(
            vertices=uvs3d,
            indices=torch.tensor(self.mano_layer.face_uvs, dtype=torch.int32),
            uvs=vertices_uv,
            uv_indices=self.mano_layer.faces.to(torch.int32),
            material=pyredner.Material(use_vertex_color=True),
            colors=vertices_uvshape
        )

        camera_uv = pyredner.Camera(
            camera_type=pyredner.camera_type.orthographic,
            position=torch.tensor([0, 0, 1.], dtype=torch.float32),
            look_at=torch.tensor([0, 0, 0.], dtype=torch.float32),
            up=torch.tensor([0, 1., 0], dtype=torch.float32),
            resolution=self.mano_layer.tex_diffuse_mean.shape[:2],
        )

        g_buffer = pyredner.render_g_buffer(
            scene=pyredner.Scene(camera=camera_uv, objects=[mano_uv_front]),
            channels=[pyredner.channels.vertex_color],
        )
        vertex_color = g_buffer[..., :3]

        return torch.where(vertex_color > 0, 1., 0.)

    def optimize_texture(self, texture, mask, n_iterations=300, learning_rate=1e-5, lambda_reg_tex=1e3):
        """NIMBLEを用いてテクスチャを最適化します。"""
        coeffs_tex = torch.zeros(10, dtype=torch.float32, requires_grad=True, device=pyredner.get_device())
        optimizer = torch.optim.SGD([coeffs_tex], lr=learning_rate)

        target_texture = torch.pow(texture, 2.2).detach()
        mask = mask.detach()

        pp = ProgressPlot(line_names=['loss_mse', 'reg_tex'])
        for _ in range(n_iterations):
            optimizer.zero_grad()
            diffuse_reflectance = torch.sum(
                coeffs_tex * self.mano_layer.tex_diffuse_basis.to(pyredner.get_device()), dim=-1)
            diffuse_reflectance += self.mano_layer.tex_diffuse_mean.to(pyredner.get_device())

            loss_mse = torch.pow((diffuse_reflectance - target_texture) * mask, 2).sum()
            reg_tex = torch.pow(coeffs_tex, 2).sum() * lambda_reg_tex
            loss = loss_mse + reg_tex
            loss.backward()
            optimizer.step()
            pp.update([[loss_mse.item(), reg_tex.item()]])
        pp.finalize()

        return torch.clamp(torch.pow(diffuse_reflectance, 1 / 2.2), 0, 1), coeffs_tex

    def project_3D_points(self, cam_mat, pts3D, is_OpenGL_coords=True):
        """3Dポイントを2D画像座標に投影します。"""
        if not isinstance(pts3D, np.ndarray):
            pts3D = pts3D.squeeze(0).detach().cpu().numpy()
        else:
            pts3D = pts3D.squeeze(axis=0)
        assert pts3D.shape[-1] == 3
        assert len(pts3D.shape) == 2

        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        if is_OpenGL_coords:
            pts3D = pts3D.dot(coord_change_mat.T)

        proj_pts = pts3D.dot(cam_mat.T)
        proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]], axis=1)

        assert len(proj_pts.shape) == 2

        return proj_pts

    def showHandJoints(self, imgInOrg, gtIn, filename=None):
        """手のジョイントとボーンを画像上に描画します。"""
        imgIn = np.copy(imgInOrg)

        # 指ごとに色を設定
        joint_color_code = [[139, 53, 255],
                            [0, 56, 255],
                            [43, 140, 237],
                            [37, 168, 36],
                            [147, 147, 0],
                            [70, 17, 145]]

        limbs = [[0, 1],[1, 2],[2, 3],[3,17],[0, 4],[4, 5],[5, 6],[6, 18],[0, 7],[7, 8],[8, 9],[9, 19],
                 [0, 10],[10, 11],[11, 12],[12, 20],[0, 13],[13, 14],[14, 15],[15, 16]]

        gtIn = np.round(gtIn).astype(np.int32)

        for joint_num in range(gtIn.shape[0]):
            color_code_num = (joint_num // 4)
            joint_color = [c + 35 * (joint_num % 4) for c in joint_color_code[color_code_num]]
            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):
            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if 5 < length < 150:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                limb_color = [c + 35 * (limb_num % 4) for c in joint_color_code[color_code_num]]
                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

        if filename is not None:
            cv2.imwrite(filename, imgIn)

        return imgIn

    def render_skeleton(self, num):
        """手のスケルトンをレンダリングし、画像として保存します。"""
        joints_3d = self.gen_hand_joints_trans

        # カメラパラメータの取得
        K = self.K.detach().cpu().numpy()

        # ジョイントの投影（3D -> 2D）
        joints_2d = self.project_3D_points(K, joints_3d)

        # 画像の解像度に合わせてスケール調整
        joints_2d[:, 0] = joints_2d[:, 0]
        joints_2d[:, 1] = self.resolution[0] - joints_2d[:, 1]

        # スケルトンの描画
        skeleton_image = self.showHandJoints(self.background_image, joints_2d)

        # 反転
        skeleton_image = cv2.flip(skeleton_image, 0)

        # リサイズとパディング
        # skeleton_image_uint8 = (skeleton_image * 255).astype(np.uint8)
        skeleton_image_resized = resize_and_pad_image(skeleton_image)

        # 画像を保存
        output_dir = 'test_data_ad/skeleton'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'hand_skeleton{num}.png')
        # cv2.imwrite(output_path, skeleton_image_resized)
        cv2.imwrite(output_path, cv2.cvtColor(skeleton_image_resized, cv2.COLOR_RGB2BGR))
        print(f"スケルトン画像を保存しました: {output_path}")

        return output_path