
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
from grabnet.tests.tester import Tester

from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu

# View in Plotly
class PlotlyFigure():
    def __init__(self):
        self.fig = go.Figure()
        self.fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    projection=dict(type='orthographic')
                )
            ),
            showlegend=True,
        )

    def add_mesh(self, verts, faces, **kwargs):
        self.fig.add_trace(
            go.Mesh3d(
                x=verts[:,0], 
                y=verts[:,1], 
                z=verts[:,2], 
                i=faces[:,0],
                j=faces[:,1],
                k=faces[:,2],
                showlegend=True,
                **kwargs)
        )
    def add_points(self, verts, **kwargs):
        self.fig.add_trace(
            go.Scatter3d(
                x=verts[:,0], 
                y=verts[:,1], 
                z=verts[:,2], 
                mode='markers',
                **kwargs)
        )

    def update_layout(self, **kwargs):
        self.fig.update_layout(**kwargs)
    
    def show(self):
        self.fig.show()