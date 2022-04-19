FROM python:3.8-slim
WORKDIR /cnn
COPY docker-requirements.txt .
RUN python -m pip install --no-cache-dir -r docker-requirements.txt
COPY /src/notebooks/inception_v3/inception_v3.ipynb src/notebooks/inception_v3/
COPY /src/notebooks/vgg16/vgg16.ipynb src/notebooks/vgg16/
COPY /src/webapp src/webapp
COPY /src/scripts src/scripts
RUN mkdir metadata data models