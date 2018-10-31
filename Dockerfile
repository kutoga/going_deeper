FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 python3-pip

WORKDIR /going_deeper
RUN mkdir -p .tmp

COPY requirements.txt .
RUN python3.6 -mpip install -r requirements.txt

RUN python3.6 -c 'from torchvision import datasets; datasets.MNIST(".tmp/data", download=True).download()' && \
    python3.6 -c 'from torchvision import datasets; datasets.CIFAR10(".tmp/data", download=True).download()'

RUN apt-get install -y vim

COPY gdeep gdeep
COPY 00_test_linear.py .

ENTRYPOINT ["python3.6", "-u", "./00_test_linear.py"]
