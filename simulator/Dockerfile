# Sensor simulator. Stands in for the hardware; not part of the product.
# Built from the project root:  docker build -f simulator/Dockerfile -t farm-simulator .
FROM python:3.12-slim
WORKDIR /app
COPY simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulator/simulate.py .
EXPOSE 8000
CMD ["python", "simulate.py"]
