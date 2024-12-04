# SE3 Diffusion

## 🛠️ Installation
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 

Then install (this might take a while).
```bash
mamba env create -f conda_environment.yaml
conda activate se3diffuser
```

Install mamba using:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
rm Miniforge3-$(uname)-$(uname -m).sh 
export PATH=$PATH:/home/$USER/miniforge3/bin
```

### 3. Prerequisites
This repo is built-off the [ARM repository](https://github.com/stepjam/ARM) by James et al. The prerequisites are the same as ARM. 
#### PyRep and Coppelia Simulator
Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz)
- [Others](https://www.coppeliarobotics.com/previousVersions#)

```bash
cd <install_dir>
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu<UBUNTU_VERSION>.tar.xz
tar -xvf CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu<UBUNTU_VERSION>.tar.xz
rm CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu<UBUNTU_VERSION>.tar.xz
```

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

#### RLBench

PerAct uses my [RLBench fork](https://github.com/beautifulv0id/RLBench.git). 

```bash
cd <install_dir>
git clone -b peract git@github.com:beautifulv0id/RLBench.git
cd RLBench
pip install -r requirements.txt
python setup.py develop
```

## Download Dataset
Peract provides [pre-generated RLBench demonstrations](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ&usp=share_link) on google drive.

Download is recommended through `rclone` with Google API Console enabled. A detailed description how to setup the remote access is provided [here](https://rclone.org/drive/). 

Note: You need to share the files with your drive.

Use `scripts/download_peract.sh` to download tasks from the drive.

Bring dataset in our format:
```bash
cd data_preprocessing
python rearrange_rlbench_demos.py