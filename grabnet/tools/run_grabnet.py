import os
import numpy as np
import torch
from grabnet.tools.load_mesh import load_obj_verts_

import trimesh
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import plotly.graph_objects as go
import numpy as np
import torch
import os

from grabnet.tools.meshviewer import Mesh as M
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
# from grabnet.tests.tester import Tester

from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu


def run_grabnet(
    obj_path, grabnet, bps, rh_model, coarse_net, refine_net,
    rotmat=None, trans_rhand=None, global_orent_rhand_rotmat=None
):
    """
    Runs the GrabNet algorithm on the given object.

    Args:
        obj_path (str): The path to the object file.
        grabnet: The GrabNet model.
        bps: The BPS model.
        rh_model: The RH model.
        coarse_net: The coarse network.
        refine_net: The refine network.
        rotmat (torch.Tensor, optional): The rotation matrix. Defaults to None.
        trans_rhand (torch.Tensor, optional): The translation vector for the right hand. Defaults to None.
        global_orent_rhand_rotmat (torch.Tensor, optional): The global orientation rotation matrix for the right hand. Defaults to None.

    Returns:
        tuple: A tuple containing the mesh object, the generated vertices from the coarse network, and the generated vertices from the refine network.
    """
    verts_obj, mesh_obj, re_scale = load_obj_verts_(obj_path, rotmat=rotmat, scale=1., n_sample_verts=10000)
    bps_object = bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']
    bps_object = bps_object.to(grabnet.device)
    verts_object = torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0)
    obj_name = os.path.basename(obj_path)

    # mano params
    if trans_rhand is None:
        trans_rhand = torch.zeros(1, 3).to(grabnet.device) 
    else: 
        trans_rhand = torch.tensor(trans_rhand).to(grabnet.device)
    
    if global_orent_rhand_rotmat is None:
        global_orient_rhand_rotmat = torch.eye(3).unsqueeze(0).to(grabnet.device) 
    else: 
        global_orient_rhand_rotmat = torch.tensor(global_orent_rhand_rotmat).to(grabnet.device)

    with torch.no_grad():
        drec_cnet = coarse_net(bps_object, trans_rhand, global_orient_rhand_rotmat)
        verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

        # model = rh_model(**drec_cnet)

        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, verts_object.to(grabnet.device))

        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = verts_object.to(grabnet.device)
        drec_cnet['h2o_dist'] = h2o.abs()

        drec_rnet = refine_net(**drec_cnet)
        rh_gen_rnet = rh_model(**drec_rnet, return_tips=True)
        # rh_gen_rnet = rh_model(**drec_rnet)
        verts_rh_gen_rnet = rh_gen_rnet.vertices
        try:
            print("joint shape is ", rh_gen_rnet.joints.shape)
        except:
            print("Error")
        
    return mesh_obj, verts_rh_gen_cnet, verts_rh_gen_rnet, rh_gen_rnet