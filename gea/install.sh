# #! /bin/sh
git lfs install
git lfs track "*.plugin"
mkdir -p ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -O ~/.mujoco/mujoco.zip
unzip ~/.mujoco/mujoco.zip -d ~/.mujoco/
rm ~/.mujoco/mujoco.zip
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco.tar.gz
tar -xvf ~/.mujoco/mujoco.tar.gz -C ~/.mujoco/
wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt
pip install lockfile
pip install -e d4rl/
pip install -e mjrl/
pip install -e metaworld/
pip install robosuite/
pip install -e doodad/
pip install -e mopa-rl/
pip install -e rlkit/
pip install -r requirements.txt
pip install dm-env
pip install distracting-control
pip install dm-control==1.0.12
pip install mujoco==2.3.5
pip install numpy==1.23.5
pip install mujoco-py==2.0.2.5
pip install hydra-core==1.3.2
pip install hydra-submitit-launcher==1.2.0
pip install wandb
pip install --upgrade networkx # for removing annoying warning
pip install -e .
pip install PyOpenGL==3.1.7 PyOpenGL_accelerate==3.1.7
pip install cython==0.29.21
pip install imageio==2.34.0
pip install imageio-ffmpeg==0.4.9
pip install open3d
pip install plantcv
pip install urdfpy
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
pip install -e .
