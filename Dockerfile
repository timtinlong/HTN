FROM ubuntu:18.04
#MAINTAINER Timothy Yu "timothy_yu@sfu.ca"
RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev
#We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==1.15.5
RUN pip3 install flask loguru jsonpickle opencv-python==4.5.1.48 imageio==2.9.0 imageio-ffmpeg==0.4.3 tf-slim==1.1.0 Pillow==8.1.0 h5py==2.10.0


COPY . /app
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]