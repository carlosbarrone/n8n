
FROM node:22-bookworm-slim AS node-runtime

FROM n8nio/runners:1.121.0 AS stock-runners

FROM python:3.13-slim-bookworm

RUN apt-get update \
	&& apt-get install -y --no-install-recommends tini ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
	&& ln -sf /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
	&& ln -sf /usr/local/lib/node_modules/corepack/dist/pnpm.js /usr/local/bin/pnpm

COPY --from=stock-runners /usr/local/bin/task-runner-launcher /usr/local/bin/task-runner-launcher
COPY --from=stock-runners /opt/runners /opt/runners

RUN useradd --create-home --home-dir /home/runner --shell /bin/sh runner \
	&& mkdir -p /app \
	&& chown -R runner:runner /app /home/runner /opt/runners

COPY n8n/requirements.txt /tmp/requirements.txt

RUN cd /opt/runners/task-runner-javascript \
	&& pnpm add moment uuid

RUN python -m venv /opt/runners/task-runner-python/.venv \
	&& /opt/runners/task-runner-python/.venv/bin/pip install --no-cache-dir --upgrade pip \
	&& /opt/runners/task-runner-python/.venv/bin/pip install --no-cache-dir websockets>=15.0.1 \
	&& /opt/runners/task-runner-python/.venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

COPY n8n/n8n-task-runners.json /etc/n8n-task-runners.json

USER runner
WORKDIR /app
ENTRYPOINT ["tini", "--", "/usr/local/bin/task-runner-launcher"]
CMD ["javascript", "python"]
