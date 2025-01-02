# Install conda packages
apt-get update
apt-get install -y cuda-12-4
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc


export PATH=/usr/local/cuda-12.4/bin:$PATH
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# conda install pytorch3d -c pytorch3d
apt-get install -y nvidia-container-toolkit
apt --fix-broken install

