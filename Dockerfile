FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY perceptron_and_train.py perceptron_and_inference.py perceptron_and_harness.py ./

ENTRYPOINT ["python"]
CMD ["perceptron_and_train.py"]
