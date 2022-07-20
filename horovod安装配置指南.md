## nvidia驱动安装
### 卸载旧驱动
找到旧版本安装包
```
sudo bash NVIDIA-Linux-x86_64-450.80.02.run --uninstall
```
或者
```
cd /usr/bin
sudo ./nvidia-uninstall
```
完成后重启, 安装新版本驱动
```
sudo bash cuda_11.7.0_515.43.04_linux.run
```

## cuda 安装
```
sudo bash cuda_11.3.1_465.19.01_linux.run
# 注意这是cuda11.3的安装包(CUDA Toolkit 11.3 Update 1 Downloads)
```
安装完成后要添加环境变量
```
# 临时
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# bashrc
vim ~/.bashrc
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc

# old 
# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
# export PATH=${CUDA_HOME}/bin:${PATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```
额外包
```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev
```

## cudnn安装
```
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz

sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# cuda 10.0
# sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
# sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
# sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## nccl 安装
```
# 卸载旧版本(purge或者autoremove)
# sudo apt purge libnccl2 libnccl-dev
sudo apt remove libnccl2 libnccl-dev --autoremove
apt list | grep nccl
sudo apt purge nccl-local-repo-ubuntu1804-2.12.12-cuda11.0

sudo dpkg -i nccl-local-repo-ubuntu1804-2.13.4-cuda10.2_1.0-1_amd64.deb
sudo cp /var/nccl-local-repo-ubuntu1804-2.13.4-cuda10.2/nccl-local-2CE49A15-keyring.gpg /usr/share/keyrings
sudo apt update
sudo apt install libnccl2 libnccl-dev
```

## 解决/usr/lib/xorg/Xorg占用gpu显存的问题
编辑如下文件，如果没有则手动添加
```
sudo vi /etc/X11/xorg.conf
```
插入以下内容，其中BusId为自己的集成显卡id，可以通过 `lspci | grep VGA`查看
```
Section "Device"
    Identifier      "intel"
    Driver          "intel"
    BusId           "PCI:0:2:0"
EndSection

Section "Screen"
    Identifier      "intel"
    Device          "intel"
EndSection
```

## 安装 conda 环境
[Build a Conda Environment with GPU Support for Horovod](https://horovod.readthedocs.io/en/stable/conda.html)
文件`environment.yml`
```
name: venv
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
- bokeh
- cmake  # insures that Gloo library extensions will be built
- cudnn
- cupti
- cxx-compiler  # insures C and C++ compilers are available
- mpi4py # installs cuda-aware openmpi
- nccl
- nodejs
- nvcc_linux-64=11.3 # configures environment to be "cuda-aware"
- cudatoolkit=11.3
- pip
- python=3.7
```

```
conda env create -f environment.yml
```

出现提示`Version of installed CUDA didn't match package`
conda安装的`nvcc_linux-64`的版本需要和系统安装的cuda版本匹配(不是`cudatoolkit`)
```
# nvcc查看cuda版本
# nvcc -V
nvcc --version
# 安装对应版本nvcc_linux-64
conda install nvcc_linux-64=11.3
```

## tensorflow和pytorch安装
```
pip install tensorflow-gpu==1.15.2

# 低版本tensorflow需要使用低版本的protobuf
pip install protobuf==3.20

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
conda安装`pytorch`需要额外channel: `-c conda-forge`

## horovod安装
### 不使用nccl
```
环境变量中并未配置cuda_home
export HOROVOD_CUDA_HOME=$CUDA_HOME
# export HOROVOD_GPU_OPERATIONS=NCCL

HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod

horovodrun --check-build

Available Frameworks:
    [X] TensorFlow
    [ ] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [ ] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```
没有`export HOROVOD_CUDA_HOME=$CUDA_HOME`会报错：
```
horovod could not find cudatoolkit (missing: cudatoolkit_include_dir)
```

### 使用nccl
```
export ENV_PREFIX=/home/psdz/miniconda3/envs/venv
export HOROVOD_CUDA_HOME=/usr/local/cuda
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL

HOROVOD_WITH_NCCL=1 HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod

Available Frameworks:
    [X] TensorFlow
    [ ] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```

### check the installation
```
horovodrun --check-build
```

如果输入horovod[tensorflow], 那么horovod会自动下载最新版本的tensorflow
```
# 错误：
HOROVOD_WITH_NCCL=1 HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod[tensorflow]
```

高版本torch
```
Changing
inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
to
inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
seems to work.
```

## pip 百度源
```
https://mirror.baidu.com/pypi/simple
```


HOROVOD_CUDA_HOME=/usr/local/cuda HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]==0.20 --no-cache-dir



pip install protobuf==3.20

```
HOROVOD_WITH_NCCL=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 pip install --no-cache-dir horovod
```
如果输入horovod[tensorflow], 那么horovod会自动下载最新版本的tensorflow
```
错误：
HOROVOD_WITH_NCCL=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 pip install --no-cache-dir horovod[tensorflow]
```