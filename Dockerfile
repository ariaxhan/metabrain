# metabrain MCP server, stdio transport, for the Docker MCP catalog.
#
# The memory database lives on the /data volume so it survives the container.
# Override the path with METABRAIN_DB; the default is /data/agent.db.

FROM python:3.12-slim AS build
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /wheels

FROM python:3.12-slim
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir "$(echo /wheels/*.whl)[mcp]" \
    && pip cache purge || true

RUN useradd --create-home --uid 10001 mcp \
    && mkdir -p /data \
    && chown mcp:mcp /data

ENV METABRAIN_DB=/data/agent.db
VOLUME ["/data"]

USER mcp
WORKDIR /data

ENTRYPOINT ["sh", "-c", "exec metabrain-mcp --db \"$METABRAIN_DB\""]
