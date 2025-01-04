import numpy as np
import pickle
import trimesh
from psbody.mesh.colors import name_to_rgb
from psbody.mesh import Mesh

from grabnet.tools.meshviewer import Mesh as M
from grabnet.tools.vis_tools import points_to_spheres


def load_obj_verts_(mesh_path, rotmat=None, scale=1., n_sample_verts=10000):
    """
    Load object vertices from an OBJ file and perform preprocessing steps.

    Args:
        mesh_path (str): The path to the OBJ file.
        scale (float, optional): Scaling factor for the object. Defaults to 1.
        n_sample_verts (int, optional): Number of vertices to sample. Defaults to 10000.

    Returns:
        verts_sampled: the sampled vertices
        obj_mesh: the modified mesh object.
    """

    obj_trimesh = trimesh.load(mesh_path)
    obj_mesh = Mesh(v=obj_trimesh.vertices, f = obj_trimesh.faces, vc=name_to_rgb['green'])

    obj_mesh.reset_normals()
    obj_mesh.vc = obj_mesh.colors_like('green')

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.v, axis=1).max()
    if  max_length > .3:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.v = obj_mesh.v/re_scale
    else:
        re_scale = 1.0

    # object_fullpts = obj_mesh.v
    # maximum = object_fullpts.max(0, keepdims=True)
    # minimum = object_fullpts.min(0, keepdims=True)

    # offset = ( maximum + minimum) / 2
    # verts_obj = object_fullpts - offset
    # obj_mesh.v = verts_obj

    if rotmat is not None:
        obj_mesh.rotate_vertices(rotmat)

    while (obj_mesh.v.shape[0] < n_sample_verts):
        mesh = M(vertices=obj_mesh.v, faces = obj_mesh.f)
        mesh = mesh.subdivide()
        obj_mesh = Mesh(v=mesh.vertices, f = mesh.faces, vc=name_to_rgb['green'])
        print("Subdivide. obj_mesh.v.shape:", obj_mesh.v.shape)

    verts_obj = obj_mesh.v
    verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, obj_mesh, re_scale

def load_ho_meta(
        meta_fp = "data/HO3D_v3/train/ABF10/meta/0000.pkl"
    ):
    with open(meta_fp, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    anno = {}
    for k, v in meta.items():
        if v is None or v == 'None':
            anno[k] = None
        else:
            anno[k] = np.array(v)
    return anno
