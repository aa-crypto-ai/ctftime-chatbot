# pick the one matching your GPU spec - https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

RUN apt update && \
    apt install -y bash \
                   vim \
                   build-essential \
                   git \
                   curl \
                   wget \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user

RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O ./anaconda.sh && bash anaconda.sh -b
ENV PATH=/home/user/anaconda3/envs/llm-rag/bin:/home/user/anaconda3/bin:$PATH

COPY . /usr/src/app/
WORKDIR /usr/src/app/

RUN conda env create -f environment.yml

CMD ["python", "app.py"]