# gateway service.
# Built from the project root:  docker build -f services/gateway/Dockerfile -t farm-gateway .
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first. Docker caches this layer, so editing a .py file
# does not trigger a re-install.
COPY services/gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/gateway/app ./app

EXPOSE 8000
CMD ["python", "-m", "app.main"]
