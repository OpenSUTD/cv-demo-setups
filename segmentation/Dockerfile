# Latest NGC TensorFlow CUDA9.0 Image
FROM nvcr.io/nvidia/tensorflow:18.08-py3

LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    apt-utils && \
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
    build-essential \
    libopenblas-base \
    libjpeg-dev \
    libpng-dev && \
    ldconfig && \
    apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE && \
    add-apt-repository \
    "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u && \
    apt-get update && \
    apt-get install librealsense2-dkms -y && \
    apt-get install librealsense2-utils -y && \
    apt-get install librealsense2-dev -y && \
    apt-get install librealsense2-dbg -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

EXPOSE 8080
EXPOSE 5000

WORKDIR /app

ADD requirements.txt /app/requirements.txt

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py && \
    pip install Cython && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /home/$NB_USER/.cache

COPY . /app

USER 1000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
