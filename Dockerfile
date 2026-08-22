FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY container_train_src/ container_train_src/
COPY container_inference_src/ container_inference_src/
COPY container_harness_src/ container_harness_src/

ENTRYPOINT ["python"]
CMD ["container_train_src/perceptron_and_train.py"]
