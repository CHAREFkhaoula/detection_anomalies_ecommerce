FROM python:3.9-slim
RUN pip install pandas
RUN pip install kafka-python
WORKDIR /app
COPY app/ .
CMD ["python", "producer.py"]