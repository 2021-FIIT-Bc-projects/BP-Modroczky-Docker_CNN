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
CMD gunicorn --threads $THREADS --worker-class gthread --bind 0.0.0.0:$PORT classifier:app
EXPOSE $PORT