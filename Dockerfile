FROM python:3.12.3
EXPOSE 5000
WORKDIR /app    
RUN apt-get update && apt-get install -y \ 
    tesseract-ocr \
    python3-opencv \
    libtesseract-dev
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["flask", "run", "--host","0.0.0.0"]