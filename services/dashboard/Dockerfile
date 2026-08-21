# dashboard service.
# Built from the project root:  docker build -f services/dashboard/Dockerfile -t farm-dashboard .
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first. Docker caches this layer, so editing a .py file
# does not trigger a re-install.
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/dashboard/app ./app
COPY services/dashboard/static ./static

EXPOSE 8000
CMD ["python", "-m", "app.main"]
