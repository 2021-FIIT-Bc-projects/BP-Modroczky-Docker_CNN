FROM python:3.8-slim
WORKDIR /cnn
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r docker-requirements.txt
COPY /src src
RUN mkdir metadata data models logs plots