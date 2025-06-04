FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
ARG use_mirror

COPY requirements.txt /root
RUN if [ "$use_mirror" = "true" ]; then \
    sed -i 's#archive.ubuntu.com#mirrors.tuna.tsinghua.edu.cn#g' /etc/apt/sources.list; \
    fi; \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    #&& apt-get -y install python3-pip python3-venv libegl1 libgl1 libgomp1 xorg sqlite3 pkg-config cmake libcairo2-dev \
    && apt-get -y install python3-pip python3-venv libegl1 libgl1 libgomp1 xorg sqlite3 \
    && if [ "$use_mirror" = "true" ]; then \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; \
    fi; \
    python3 -m venv /root/venv \
    && source /root/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /root/requirements.txt \
    && pip cache purge \
    && echo "source /root/venv/bin/activate" >/root/.bashrc

WORKDIR /root/SRNN
