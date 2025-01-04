import numpy as np
import torch

def calc_vertex_normals(verts, faces):
    face_normals = calc_face_normals(verts, faces)
    vertex_normals = np.zeros(verts.shape)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:,None]
    return vertex_normals

def calc_face_normals(verts, faces):
    v1 = verts[faces[:,0].type(torch.long)]
    v2 = verts[faces[:,1].type(torch.long)]
    v3 = verts[faces[:,2].type(torch.long)]
    return np.cross(v2-v1, v3-v1)