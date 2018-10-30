FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 python3-pip

WORKDIR /going_deeper

COPY requirements.txt .
RUN python3.6 -mpip install -r requirements.txt

COPY gdeep gdeep
COPY 00_test_linear.py .

ENTRYPOINT ["python3.6", "./00_test_linear.py"]