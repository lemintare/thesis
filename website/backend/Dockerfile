FROM python:3.12-slim

WORKDIR /workdir

ENV PYTHONPATH=/workdir/src

RUN pip install uv 

COPY . .

RUN uv sync

CMD [ "uv", "run", "uvicorn", "src.main:app", "--port", "8888", "--host", "0.0.0.0", "--reload" ]

