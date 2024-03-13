FROM python:3.12-slim-bookworm

LABEL maintainer="Murilo Cunha <murilo.k.s.cunha95@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl unzip git
RUN curl -L -o duckdb_cli-linux-amd64.zip https://github.com/duckdb/duckdb/releases/download/v0.9.2/duckdb_cli-linux-amd64.zip \
    && unzip duckdb_cli-linux-amd64.zip -d /usr/local/bin \
    && rm duckdb_cli-linux-amd64.zip
RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash
