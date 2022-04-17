FROM python:3.8-slim
WORKDIR /cnn
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY /src src
RUN mkdir metadata data models .kaggle
WORKDIR /cnn/metadata
RUN touch obtain.json augment.json
WORKDIR /cnn/.kaggle
RUN touch kaggle.json
WORKDIR /cnn/src/webapp
CMD gunicorn --timeout 90 --bind 0.0.0.0:5000 classifier:app
EXPOSE 5000