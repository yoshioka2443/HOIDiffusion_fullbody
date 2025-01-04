import torch
from scipy.spatial.transform import Rotation as R


def apply_transform_to_mesh(verts, anno):
    objRmat = torch.tensor(R.from_rotvec(anno['objRot'][:, 0]).as_matrix(), dtype=torch.float32).to(verts.device)
    objTrans = torch.tensor(anno['objTrans'], dtype=torch.float32).to(verts.device)
    # verts = (objRmat @ verts.T + objTrans.T).T
    print(verts.shape, objRmat.shape, objTrans.shape)
    verts = verts @ objRmat.T + objTrans
    return verts

def apply_trans(verts, rot=None, trans=None):
    objRmat = rot
    objTrans = trans
    # verts = (objRmat @ verts.T + objTrans.T).T
    print(verts.shape, objRmat.shape, objTrans.shape)
    verts = verts @ objRmat.T + objTrans
    return verts