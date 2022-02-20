FROM deepnote/python:3.7

RUN apt update && apt install -y ffmpeg libsm6 libxext6 libgl1-mesa-dev