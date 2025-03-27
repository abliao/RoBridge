eval "$(conda shell.bash hook)" # necesssary to activate conda envs
conda activate $1

cd ../doodad
pip install -r requirements.txt
pip install -e .
cd ../robosuite
pip install -r requirements-extra.txt
pip install -e requirements.txt
pip install -e .
cd ../viskit
pip install -e .
cd ../rlkit
pip install -r requirements.txt
pip install -e .


# ./install-ompl-ubuntu.sh --python # NOTE: this will take awhile to run
# echo "/home/mdalal/ompl/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/$1/lib/python3.8/site-packages/ompl.pth
