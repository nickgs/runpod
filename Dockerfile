FROM python:3.10-slim

WORKDIR /
COPY requirements.txt /
COPY UIGen.py /
RUN pip install -r requirements.txt --no-cache-dir runpod 

# Start the container
CMD ["python3", "-u", "sentiment_analysis.py"]
