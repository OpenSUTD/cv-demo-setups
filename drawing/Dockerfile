# OpenPose doc recommends
# CUDA 8.0 + cuDNN 5
FROM tlkh/oh19-cv-demo:openpose-base

LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"

USER root

ENV DEBIAN_FRONTEND noninteractive

COPY . /app

USER 1000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
