export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL='egl'
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python gea/launch_scripts/metaworld.py
