FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile, in their own layer, so editing the app
# does not reinstall them. --no-dev leaves the notebook toolchain out of the image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app.py index.html ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8100

USER app
EXPOSE 8100

# Shell form on purpose: hosts that inject $PORT (Render, Railway, Fly, Cloud Run)
# need it expanded at start time. `exec` hands PID 1 to uvicorn so SIGTERM still
# reaches it and shutdown stays graceful.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
