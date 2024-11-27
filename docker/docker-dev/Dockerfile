# FROM paddleocr:cpu
# FROM python:3.9
FROM llama_factory:0.0.1
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="zhangjl19"
LABEL version = "dev-1.0.0"
LABEL description = "llama_factory dev environments"

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    echo 'Asia/Shanghai' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y git && \
    apt-get install -y openssh-server && \
    apt-get install -y pdsh && \
    apt-get install -y net-tools && \
    apt-get install -y iputils-ping && \
    apt-get install -y vim && \
    pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]
