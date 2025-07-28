FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY adobe_hackathon_ai.py ./

ENTRYPOINT ["python", "adobe_hackathon_ai.py"]
