FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.14.0 \
    torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY optimization/ ./optimization/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]