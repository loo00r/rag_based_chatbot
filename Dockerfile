FROM python:3.12-slim

WORKDIR /project

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ app/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]
