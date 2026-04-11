FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r api/requirements.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]