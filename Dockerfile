FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements_assistant.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_assistant.txt

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
