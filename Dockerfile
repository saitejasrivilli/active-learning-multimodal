FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt ||     pip install --no-cache-dir torch transformers datasets scikit-learn pandas matplotlib numpy tqdm

COPY . .

RUN mkdir -p results data

CMD ["python3", "lightweight_al.py"]
