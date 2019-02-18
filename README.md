# openhouse-demo-cv

SUTD Open House 2019 CV Demo

## Installation

Assuming Ubuntu 18.04 or Ubuntu 16.04

### 1. Intel RealSense Setup

```
sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# Ubuntu 16.04
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# Ubuntu 18.04
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u

sudo apt-get install librealsense2-dkms -y
sudo apt-get install librealsense2-utils -y
sudo apt-get install librealsense2-dev -y
sudo apt-get install librealsense2-dbg -y

# run realsense-viewer to verify
```

### 2. Demo setup

```
WIP
```

## Running the Demo

```
xhost +
docker run --rm --privileged \
 --net=host --ipc=host \
 --env="DISPLAY" \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --device=/dev/dri:/dev/dri \
 -v /dev/video/ \
 tlkh/oh-cv-demo
```

## Acknowledgements

Code adapted from [`matterport/Mask_RCNN`](https://github.com/matterport/Mask_RCNN)




