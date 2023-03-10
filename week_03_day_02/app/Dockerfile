FROM bitnami/pytorch

ENV PORT=8000

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt --default-timeout=3600

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
