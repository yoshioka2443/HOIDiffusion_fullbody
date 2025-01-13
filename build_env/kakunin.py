import sys
python_version = sys.version
import numpy as np
numpy_version = np.__version__
print('--- Python Information ---')
print('Python version:', python_version)
print('NumPy version:', numpy_version)

import torch
print('--- PyTorch Information ---')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device count:', torch.cuda.device_count())
print('----------------------------\n')

try:
    import pytorch3d
    print('--- PyTorch3D Information ---')
    print('PyTorch3D version:', pytorch3d.__version__)
except ImportError:
    print('PyTorch3D is not installed.')

try:
    import pyrender
    print('--- Pyrender Information ---')
    print('Pyrender version:', pyrender.__version__)
except ImportError:
    print('Pyrender is not installed.')

try:
    import pyredner
    print('--- Pyredner Information ---')
    print('Pyredner version:', pyredner)
except ImportError:
    print('Pyredner is not installed.')