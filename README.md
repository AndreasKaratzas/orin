
# Setting up the NVIDIA Jetson Orin Nano

This tutorial was written for the NVIDIA Jetson Orin Nano Developer Kit (8 GB). The tutorial was written on `2024-2-13:19-42-5`. Details of the board are:

| Key | Value |
| --- | --- |
| P-Number | p3767-0005 |
| Module | NVIDIA Jetson Orin Nano (Developer kit) |
| SoC | tegra234 |
| CUDA Arch BIN | 8.7 |
| L4T | 36.2.0 |
| Jetpack | 6.0 DP |
| Machine | aarch64 |
| System | Linux |
| Distribution | Ubuntu 22.04 Jammy Jellyfish |
| Release | 5.15.122-tegra |
| Python | 3.11.5 |
| CUDA | 12.2.140 |
| OpenCV | 4.8.0 |
| OpenCV-Cuda | False |
| cuDNN | 8.9.4.25 |
| TensorRT | 8.6.2.3 |
| VPI | 3.0.10 |
| Vulkan | 1.3.204 |

__IMPORTANT WARNING__: The host must be running on __Ubuntu 20.04__ to flash the board. 


### Flashing the Board


The board can be flashed using the SDK Manager. The SDK Manager can be downloaded from the [NVIDIA website](https://developer.nvidia.com/nvidia-sdk-manager). To flash the board, force the board into recovery mode by following the steps in [JetsonHacks Tutorial](https://www.youtube.com/watch?v=q4fGac-nrTI&t=218s). Then, select the following options:

- Host Machine: Ubuntu 20.04
- Target Hardware: Jetson Orin Nano Developer Kit 8 GB
- Target OS: Jetpack 6.0 DP
- DeepStream: 6.0

In my case, I had already mounted an `NVME SSD` as a storage component. Most tutorials do SD cards, but I went with a mainstream 1 TB NVME SSD for performance reasons. Select the `pre-config` option to set the board and initialize it with a username and a password. For simplicity, the username and password are set to `nvidia`. The board will be flashed, and it will be ready to use.


### Setting up the Board

First, update the system:
```bash
sudo apt-get update
sudo apt-get upgrade
sudo reboot
```

Then, install the following packages:
```bash
sudo apt-get upgrade
sudo apt-get install python3 python3-dev python3-distutils python3-venv python3-pip
sudo apt-get install ssh firefox zlib1g software-properties-common lsb-release cmake build-essential libtool autoconf unzip wget htop ninja-build terminator
```

These packages will prepare the board for development. If you are working extensively with Python, install `conda` package manager. The following commands will install `miniconda`:
```bash
cd Downloads/
sudo apt upgrade
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
```


### Checking CUDA and cuDNN

Before we advance on checking CUDA and cuDNN, we need to verify `gcc` and `nvidia-smi`:
```bash
gcc --version
nvidia-smi
```

We will begin with checking CUDA:
```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery/
make
./deviceQuery
```

After checking CUDA, we will check cuDNN:
```bash
cp -r /usr/src/cudnn_samples_v8/ ~/Documents/workspace/
cd cudnn_samples_v8/mnistCUDNN/
sudo apt install libfreeimage3 libfreeimage-dev
sudo apt-get update
sudo apt-get upgrade
make clean && make
./mnistCUDNN
```


### Monitoring the Board

To monitor the board, we will install `jetson-stats`:
```bash
sudo pip3 install -U jetson-stats
```

To check your board details and version of different software using `jtop`, as well as the usage across its computing resources and power consumption, there are some Python scripts that use `jtop`. For example [`jtop_properties.py`](https://github.com/rbonghi/jetson_stats/blob/master/examples/jtop_properties.py) is a quick way to monitor the aforementioned.


### VS Code

If you are a Visual Studio Code user, it is supported on the Jetson. Run the following commands:

```bash
cd Downloads/
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt install apt-transport-https
sudo apt update
sudo apt install code
```


### Case

For my board, I bought the [Yahboom CUBE nano case](https://www.amazon.com/Yahboom-Dissipation-Protect-Cooling-Antenna/dp/B0CD71X8SV). On [their page](http://www.yahboom.net/study/CUBE_NANO), there are also [tutorials](https://youtu.be/anbMcWsagn8) and [code](https://drive.google.com/drive/folders/1A4L1ec-Na1_K0K1LXdnzSCva2iZ02YVX) for setting up the case and configuring the OLED screen that comes with it. Finally, there is also a [GitHub repo](https://github.com/YahboomTechnology/Jetson-CUBE-case) associated with the case.


### Install PyTorch

First, we will install the dependencies:
```bash
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```

Then, we initialize the conda environment for PyTorch:
```bash
conda create -n torchenv python=3.10 pip
conda activate torchenv
```

Now, we can install PyTorch:
```bash
pip install Cython
cd ~/Downloads
wget https://nvidia.box.com/shared/static/0h6tk4msrl9xz3evft9t0mpwwwkw7a32.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
pip install numpy torch-2.1.0-cp310-cp310-linux_aarch64.whl
```

Finally, we install `torchvision`:
```bash
cd ~/Downloads
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
export BUILD_VERSION=0.16.1
cd torchvision/
python setup.py install --user
cd ../
pip install Pillow
```

To test PyTorch, run the following:
```python
import torch

print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))

a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
```

To test `torchvision`, run the following:
```python
import torch
import torchvision

print(torchvision.__version__)

from torchvision.models import resnet50

m = resnet50(weights=None)
m.eval()
x = torch.randn((4,3,224,224))
m(x)
```


### Set Performance Mode

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```


### Notes

It is important to note that the NVIDIA SDK Manager must be installed on an Ubuntu 20.04 engine. I tried two different machines running Ubuntu 22.04 and attempted to flash the board, but it would yield errors. I also tried Ubuntu 18.04, but the latest supported Jetpack was `5.x.y`, and at the moment of writing, the latest Jetpack is `6.z`. Therefore, the host machine must be running Ubuntu 20.04.
