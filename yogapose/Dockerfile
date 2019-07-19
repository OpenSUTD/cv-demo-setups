FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="Timothy Liu <timothy_liu@mymail.sutd.edu.sg>"

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    apt-utils software-properties-common && \
    add-apt-repository ppa:jonathonf/gcc-7.1 && \
    apt-get update && \
    apt-get install gcc-7 g++-7 gfortran-7 -y && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    # install system packages
    software-properties-common \
    python3-dev \
    python3-tk \
    wget \
    curl \
    locales \
    ca-certificates \
    fonts-liberation \
    git \
    build-essential \
    cmake \
    # for OpenPose
    libatlas-base-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    libviennacl-dev \
    libopenblas-base \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    && ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV SHELL=/bin/bash \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py && \
    pip install --no-cache-dir Cython && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /home/$NB_USER/.cache && \
    rm -rf /tmp/*

RUN git clone https://github.com/tlkh/openpose.git && \
    cd openpose/ && mkdir build && cd build && \
    cp ../docker_files/CMakeLists.txt ../CMakeLists.txt && \
    cmake .. && make -j && cd python && make install

COPY . /app

USER 1000

ENTRYPOINT [ "python3" ]
CMD [ "demo.py" ]
