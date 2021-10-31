FROM nvcr.io/nvidia/tensorrt:20.10-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y -q apt-utils python3-pip gcc libjpeg8-dev zlib1g-dev python3-opencv

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /usr/src/yolotrt
WORKDIR /usr/src/yolotrt

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

EXPOSE 5001

CMD ["python", "app.py"]