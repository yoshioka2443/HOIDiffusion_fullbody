# conda create -n fullbody python=3.9 -y
# conda init
# conda activate fullbody

apt-get update
apt-get install libboost-dev -y
apt-get install -y libtbb2 libtbb-dev

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch3d -c pytorch3d -c conda-forge
pip install -r requirements.txt
# 

