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
```

### 2. Demo setup

```
WIP
```

# run realsense-viewer to verify

## Acknowledgements

Code adapted from [`matterport/Mask_RCNN`](https://github.com/matterport/Mask_RCNN)




