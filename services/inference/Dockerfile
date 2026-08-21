# Inference Flask service.
# Built from the project root:
#   docker build -f services/inference/Dockerfile -t farm-inference:1.0 .
FROM python:3.12-slim

WORKDIR /app

# Install packages from requirements.txt, as in the Week 16 Flask practical.
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Train the model during the image build so no manual model step is required.
COPY services/inference/train.py .
COPY data ./data
RUN python train.py

COPY services/inference/app ./app

EXPOSE 8000
CMD ["python", "-m", "app.main"]
