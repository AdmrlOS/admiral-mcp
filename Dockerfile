# syntax=docker/dockerfile:1
# Hosted admrl-mcp (streamable HTTP, OAuth protected resource). See docs/HOSTED.md.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached unless pyproject/uv.lock change).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
# Bytecode is precompiled above, so the root filesystem can be read-only.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ADMRL_MCP_PORT=8080 \
    HOME=/tmp
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin admrl
COPY --from=build --chown=root:root /app/.venv /app/.venv
USER 10001:10001
WORKDIR /tmp
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["admrl-mcp-http"]
