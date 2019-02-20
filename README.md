# SUTD Open House 2019 CV Demo

An real-time instance-segmentation computer vision demonstration for SUTD Open House 2019

## Installation

This is assuming you have an Ubuntu 18.04 or Ubuntu 16.04 system with `nvidia-drivers>=384`. There is a container available with the demo pre-packaged. The CUDA Toolkit version used by the container is CUDA 9.0 (if your driver is >=384, you are good to go!)

### 1. Intel RealSense Setup

You will need to setup the Intel RealSense drivers on your system.

```
# unplug RealSense camera

sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# Ubuntu 16.04
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# Ubuntu 18.04
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u

sudo apt-get install librealsense2-dkms -y
sudo apt-get install librealsense2-utils -y
sudo apt-get install librealsense2-dev -y
sudo apt-get install librealsense2-dbg -y

# plug in RealSense camera
# you can run realsense-viewer to verify installation
```

### 2. Docker/nvidia-docker Setup

You will need Docker and the nvidia-docker2 container runtime.

Quick setup (no warranty):

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
DEBIAN_FRONTEND=noninteractive sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install docker-ce=18.06.1~ce~3-0~ubuntu -y

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo -e $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install nvidia-docker2=2.0.3+docker18.06.1-1 nvidia-container-runtime=2.0.0+docker18.06.1-1 -y

# optional:
# sudo usermod -aG docker $USER
```

## Running the Demo

```
# one time only
docker pull tlkh/oh19-cv-demo:seg-1.0

# to run the demo
xhost +
nvidia-docker run --rm --privileged \
 --net=host --ipc=host \
 --env="DISPLAY" \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --device=/dev/dri:/dev/dri \
 -v /dev/video/ \
 tlkh/oh19-cv-demo:seg-1.0
# tlkh/oh19-cv-demo:seg-1.0-lowres
```

## Acknowledgements

* Mask R-CNN code and weights are adapted from from Matterport Inc [`matterport/Mask_RCNN`](https://github.com/matterport/Mask_RCNN).
* The NGC ([NVIDIA GPU Cloud](https://www.nvidia.com/en-sg/gpu-cloud/)) TensorFlow container is used in this project. NGC containers are NVIDIA tuned, tested, certified, and maintained containers for deep learning and HPC frameworks that take full advantage of NVIDIA GPUs on supported systems, such as NVIDIA DGX products.
* The website, its software and all content found on it are provided on an “as is” and “as available” basis. Please open an issue if you encounter problems or have a feature request.

