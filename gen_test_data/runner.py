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

from grabnet.tools.run_grabnet import run_grabnet
from grabnet.tools.plotly import PlotlyFigure
from grabnet.tools.load_mesh import load_obj_verts_, load_ho_meta
from grabnet.tools.transform import apply_transform_to_mesh, apply_trans
from grabnet.tools.texture import calc_vertex_normals, calc_face_normals

from gen_test_data.relighting import deringing
from gen_test_data.transform import apply_transform, to_homogeneous_matrix
from gen_test_data.image_edit import resize_and_pad_image

import cv2

# dataset_dir = '../datasets'
dataset_dir = "/home/projects/dataset"
# dataset_dir = "/workspace/dataset"

def save_image(original_image, image_path, img=None):
    """画像を保存します。"""
    if img == "depth" or img == "seg":
        plt.show()
        plt.imsave(image_path, original_image, cmap='gray')
        plt.close()
    elif img == "skeleton":
        # 画像を保存
        # output_dir = 'test_data_ad/skeleton'
        # os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    else:
        plt.show()
        plt.imsave(image_path, original_image)
        plt.close()
    print(f"Saved image to {image_path}")

class Runner:
    def __init__(self, output_dir='sample'):
        """GrabNetと関連モデルを初期化します。"""
        self.initialize_grabnet()
        self.output_dir = output_dir

    def initialize_grabnet(self):
        """GrabNetと関連モデルを初期化します。"""
        config_path = 'grabnet/configs/grabnet_cfg.yaml'
        # mano_right_hand_model_path = 'mano_data/mano_v1_2/models/mano/MANO_RIGHT.pkl'
        mano_right_hand_model_path = 'NIMBLE/mano_v1_2/models/mano/MANO_RIGHT.pkl'
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
        image_path = dataset_dir + f'/HO3D_v3/train/{sequence_name}/rgb/{frame_number:04d}.jpg'
        self.background_image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        # self.background_envmap = pyredner.EnvironmentMap(torch.tensor(self.background_image).to(pyredner.get_device()))

    def load_object_data(self, sequence_name, frame_number, replacement_object_name):
        """オブジェクトと手のデータを読み込みます。"""
        meta_path = dataset_dir + f'/HO3D_v3/train/{sequence_name}/meta/{frame_number:04d}.pkl'
        print("meta_path", meta_path)
        with open(meta_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.original_object_path = dataset_dir + f"/models/{data['objName']}/textured_simple.obj"
        self.replacement_object_path = dataset_dir + f"/models/{replacement_object_name}/textured_simple.obj"

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
        self.replaced_object_vertices = self.replacement_object.vertices.type(torch.float32).cpu() @ self.object_rotation.T + self.object_translation
        # print("self.original_object.vertices", self.original_object.vertices.shape)
        # print("self.object_rotation.T", self.object_rotation.T.shape)
        # print("self.object_translation", self.object_translation.shape)
        # print("original_object_vertices pre", self.original_object_vertices.shape)

        # 手のモデルをロード
        self.annotation = load_ho_meta(meta_path)
        self.mano_layer = ManoLayer()
        self.mano_layer.load_textures()
        self.mano_hand = self.mano_layer(self.annotation)
        print("Object data loaded.")
    
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
            vertices=self.original_hand_vertices.type(torch.float32),
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

        self.original_object = pyredner.Object(
            vertices=self.original_object_vertices.type(torch.float32),
            indices=self.original_object.indices.type(torch.int32),
            uvs=self.original_object.uvs,
            uv_indices=self.original_object.uv_indices,
            material=self.original_object.material
        )
        print("self.original_hand_vertices", self.original_hand_vertices.shape)
        print("self.original_object_vertices", self.original_object_vertices.shape)
        self.original_scene = pyredner.Scene(
            camera=self.camera,
            objects=[self.original_hand, self.original_object]
        )
        self.light = pyredner.AmbientLight(intensity=torch.tensor([1., 1., 1.]))
        print("Original scene created.")

    def render_original_scene(self):
        """置き換え前のシーンをレンダリングします。"""
        self.original_render = pyredner.render_deferred(self.original_scene, lights=[self.light], alpha=True)
        print("Original scene rendered.")
        # return self.original_render
    
    def make_replaced_scene(self):
        uvs = torch.stack([self.mano_layer.uv[..., 0], 1 - self.mano_layer.uv[..., 1]], -1)

        # 元の手のメッシュ
        # self.replaced_hand_vertices = self.mano_hand.vertices[0].type(torch.float32).cpu().detach()

        vertex_normals = calc_vertex_normals(self.replaced_hand_vertices, self.mano_layer.faces)

        self.replaced_hand = pyredner.Object(
            vertices=self.replaced_hand_vertices,
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

        self.replaced_object = pyredner.Object(
            vertices=self.replaced_object_vertices.type(torch.float32),
            indices=self.replacement_object.indices.type(torch.int32),
            uvs=self.replacement_object.uvs,
            uv_indices=self.replacement_object.uv_indices,
            material=self.replacement_object.material
        )

        self.replaced_scene = pyredner.Scene(
            camera=self.camera,
            objects=[self.replaced_hand, self.replaced_object]
        )
        print("Replaced scene created.")

    def render_replaced_scene(self):
        """置き換え前のシーンをレンダリングします。"""
        self.replaced_render = pyredner.render_deferred(self.replaced_scene, lights=[self.light], alpha=True)
        print("Replaced scene rendered.")
        # return self.replaced_render
    
    def render_scene(self, replace="origin", env=None, tex=None):
        if replace == "origin":
            Hand = self.original_hand
            Object = self.original_object
        elif replace == "replaced":
            Hand = self.replaced_hand
            Object = self.replaced_object

        if tex == "NIMBLE":
            Hand.material=pyredner.Material(
                diffuse_reflectance = self.nimble_texture,
                # specular_reflectance = specular_reflectance,
                # normal_map = mano_layer.tex_normal_mean.to(pyredner.get_device()),
            )
        elif tex == "blended":
            Hand.material=pyredner.Material(
                diffuse_reflectance = self.blended_texture,
                # specular_reflectance = specular_reflectance,
                # normal_map = mano_layer.tex_normal_mean.to(pyredner.get_device()),
            ) 

        scene = pyredner.Scene(
            camera=self.camera,
            objects=[Hand, Object]
        )
        if env == "estimated":
            scene.envmap = self.envmap

        scene.object_material_id = torch.tensor([1, 2])
        render = pyredner.render_deferred(scene, lights=[self.light], alpha=True)
        albedo = pyredner.render_albedo(scene)

        
    

        # depth            
        # g_buffer = pyredner.render_g_buffer(
        #     scene=scene,
        #     channels=[pyredner.channels.depth]
        # )
        # depth_map = g_buffer[..., 0]

        # render = pyredner.RenderFunction.apply
        # depth_map = render(scene, 1, [pyredner.channels.depth])
        # mask_image = render(scene, 2, [pyredner.channels.segmentation])

        g_buffer = pyredner.render_g_buffer(
            scene=scene,
            channels=[pyredner.channels.depth]
        )
        depth_map = g_buffer[..., 0]
        # seg
        try:
            g_buffer = pyredner.render_g_buffer(
                scene=scene,
                channels=[pyredner.channels.object_id]
            )
            mask_image = g_buffer[..., 0]
        except:
            mask_image = torch.zeros_like(depth_map)

        try:
            texture_image = pyredner.render_texture(scene)
        except:
            texture_image = torch.zeros_like(render)
        
        print("depth_map", depth_map.max(), depth_map.min())
        print("mask_image", mask_image.max(), mask_image.min())
        print("texture_image", texture_image.max(), texture_image.min())

        # デプス値を0から1に正規化
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        depth_image = (depth_map - min_depth) / (max_depth - min_depth)

        # save
        if replace == "origin":
            self.original_render = render.cpu().detach().numpy()
            self.original_albedo = albedo.cpu().detach().numpy()
            self.original_depth = depth_image.cpu().detach().numpy()
        elif replace == "replaced":
            self.replaced_render = render.cpu().detach().numpy()
            self.replaced_albedo = albedo.cpu().detach().numpy()
            self.replaced_depth = depth_image.cpu().detach().numpy()
            self.replaced_mask = mask_image.cpu().detach().numpy()
            self.replaced_texture = texture_image.cpu().detach().numpy()
            self.render_skeleton()
            

        else:
            print("Invalid scene name.")

    def save_images(self, num):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir, "rgb")):
            os.makedirs(os.path.join(self.output_dir, "rgb"))
        if not os.path.exists(os.path.join(self.output_dir, "albedo")):
            os.makedirs(os.path.join(self.output_dir, "albedo"))
        if not os.path.exists(os.path.join(self.output_dir, "depth")):
            os.makedirs(os.path.join(self.output_dir, "depth"))
        if not os.path.exists(os.path.join(self.output_dir, "skeleton")):
            os.makedirs(os.path.join(self.output_dir, "skeleton"))
        if not os.path.exists(os.path.join(self.output_dir, "seg")):
            os.makedirs(os.path.join(self.output_dir, "seg"))
        if not os.path.exists(os.path.join(self.output_dir, "mask")):
            os.makedirs(os.path.join(self.output_dir, "mask"))
        if not os.path.exists(os.path.join(self.output_dir, "texture")):
            os.makedirs(os.path.join(self.output_dir, "texture"))

        original_image = torch.pow(torch.from_numpy(self.original_render), 1.0 / 2.2).cpu().detach().numpy()
        # print("self.replaced_render", self.replaced_render.shape)
        # save_image(original_image, self.output_dir, filename="original.png")
        # save_image(self.background_image, self.output_dir, filename="background.png")

        self.rgb_image_path = os.path.join(self.output_dir, "rgb" ,f"rgb{num}.png")
        self.albedo_image_path = os.path.join(self.output_dir, "albedo", f"albedo{num}.png")
        self.depth_image_path = os.path.join(self.output_dir, "depth", f"depth{num}.png")
        self.skeleton_image_path = os.path.join(self.output_dir, "skeleton", f"skeleton{num}.png")
        self.seg_image_path = os.path.join(self.output_dir, "seg", f"seg{num}.png")
        self.mask_image_path = os.path.join(self.output_dir, "mask", f"mask{num}.png")
        self.texture_image_path = os.path.join(self.output_dir, "texture", f"texture{num}.png")

        save_image(self.replaced_render[:,:,:3], self.rgb_image_path)
        save_image(self.replaced_albedo, self.albedo_image_path)
        save_image(self.replaced_depth, self.depth_image_path, img="depth")
        save_image(self.replaced_skeleton, self.skeleton_image_path, img="skeleton")
        save_image(self.replaced_render[:,:,-1], self.seg_image_path, img="seg")
        save_image(self.replaced_mask, self.mask_image_path, img="mask")
        save_image(self.replaced_texture, self.texture_image_path, img="texture")
        print(f"Images saved to {self.output_dir}")
    
    def get_paths(self):
        fieldnames = ['rgb', 'seg', 'depth', 'skeleton', 'albedo', 'texture']
        image_paths = [self.rgb_image_path, self.seg_image_path, self.depth_image_path, self.skeleton_image_path, self.albedo_image_path, self.texture_image_path]
        return dict(zip(fieldnames, image_paths))

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
            scene = self.original_scene,
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
        render_albedo = pyredner.render_albedo(self.original_scene, alpha=True)
        mask = render_albedo[..., -1]

        
        target = torch.pow(torch.tensor(self.background_image, device=pyredner.get_device()), 2.2)

        lambda_reg_tex = 1e2
        lambda_reg_sh = 1e2
        loss_log = []

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
            mano_shape = pyredner.Shape(
                vertices=self.original_hand.vertices,  # original_hand の頂点
                indices=self.original_hand.indices,    # original_hand のインデックス
                # uvs=torch.tensor(self.original_hand.uvs, dtype=torch.float32) if self.original_hand.uvs is not None else None,  # UV
                # uv_indices=torch.tensor(self.original_hand.uv_indices, dtype=torch.int32) if hasattr(self.original_hand, 'uv_indices') else None,
                uvs=self.original_hand.uvs.clone().detach() if self.original_hand.uvs is not None else None,  # UV
                uv_indices=self.original_hand.uv_indices.clone().detach() if hasattr(self.original_hand, 'uv_indices') else None,
                material_id=0  # マテリアルID
            )

            obj_shape = pyredner.Shape(
                vertices=self.original_object.vertices,  # original_object の頂点
                indices=self.original_object.indices,    # original_object のインデックス
                # uvs=torch.tensor(self.original_object.uvs, dtype=torch.float32) if self.original_object.uvs is not None else None,  # UV
                # uv_indices=torch.tensor(self.original_object.uv_indices, dtype=torch.int32) if hasattr(self.original_object, 'uv_indices') else None,
                uvs=self.original_object.uvs.clone().detach() if self.original_object.uvs is not None else None,  # UV
                uv_indices=self.original_object.uv_indices.clone().detach() if hasattr(self.original_object, 'uv_indices') else None,
                material_id=1  # マテリアルID
            )
            # Shapeリストの作成
            shapes = [mano_shape, obj_shape]

            scene = pyredner.Scene(camera = self.camera,
                                shapes = shapes,
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

            # if t > 0 and t % (10 ** int(math.log10(t))) == 0:
            #     pyredner.imwrite(img.cpu(), f'{self.output_dir}/iter_{t}.png')

        pp.finalize()

        envmap = pyredner.SH_reconstruct(self.deringed_coeffs_sh, self.resolution)
        self.envmap = pyredner.EnvironmentMap(envmap)
        print("Envmap estimated.")


    def replace_object(self):
        """オブジェクトを置き換え、手とオブジェクトの頂点を取得します。"""
        translation = (self.hand_translation - self.object_translation).unsqueeze(0)

        # 元の手のパラメータ
        original_hand_params = dict(
            transl=self.hand_translation.unsqueeze(0).float().to(self.grabnet.device),
            grobal_orient=self.hand_rotation.float().to(self.grabnet.device),
            hand_pose=self.hand_pose.float().to(self.grabnet.device)
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
        self.replaced_hand_vertices = apply_transform(hand_grabnet_gen, transform_matrix)
        self.replaced_object_vertices = apply_transform(obj_vertices, transform_matrix)

        self.gen_hand_joints_trans = apply_transform(self.gen_hand.joints[:, :, :][0].cpu() + self.object_translation, transform_matrix)


        plot_mesh([self.replaced_hand_vertices, self.replaced_object_vertices], [self.mano_layer.faces.cpu(), self.replacement_object.indices.cpu()], ["hand", "obj"])
        

        # return self.replaced_mano_vertices, self.replaced_object_vertices
    
    def estimate_nimble(self):
        # テクスチャの投影と最適化
        self.project_texture_to_uv(self.background_image)
        self.compute_front_mask()
        self.projected_texture_masked = self.projected_texture * self.computed_front_mask + torch.ones_like(self.projected_texture) * (1 - self.computed_front_mask)
        self.nimble_texture, _ = self.optimize_texture(self.projected_texture_masked, self.computed_front_mask, n_iterations=100, learning_rate=1e-5, lambda_reg_tex=1e3)

        # テクスチャのブレンド
        self.blended_texture = self.computed_front_mask * self.projected_texture + (1 - self.computed_front_mask) * self.nimble_texture

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

        self.projected_texture =  pyredner.render_albedo(pyredner.Scene(camera=camera_uv, objects=[mano_uv_renderer]))

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

        self.computed_front_mask = torch.where(vertex_color > 0, 1., 0.)

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

    # skeleton
    def render_skeleton(self):
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
        self.replaced_skeleton = skeleton_image_resized

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


def plot_mesh(verts, faces, names):
    fig = PlotlyFigure()
    fig.add_mesh(verts[0], faces[0], name=names[0], opacity=0.5)
    fig.add_mesh(verts[1], faces[1], name=names[1], opacity=0.5)
    fig.show()
