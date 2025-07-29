FROM python:3.12.1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
# streamlit run app.py --server.address 0.0.0.0 --server.port 8502
# docker build -t moussasamina/arthur-dl-project .
