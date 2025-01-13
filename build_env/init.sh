apt-get update
apt-get install -y wget
apt-get install -y git
apt-get install -y unzip

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/root/miniconda3/bin:$PATH"

source ~/.bashrc

conda info -e
conda create -n fullbody python=3.10 -y
conda init bash
source ~/.bashrc
conda activate HOIDiffusion





