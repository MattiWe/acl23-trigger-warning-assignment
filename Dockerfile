FROM python:3.11.3-slim-bullseye

ARG POETRY_ENV UID=1000 GID=1000

ENV POETRY_ENV=${POETRY_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.4.2

COPY pyproject.toml poetry.lock README.md /
COPY trigger_multilabel_classification/ /trigger_multilabel_classification/

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

WORKDIR /trigger_multilabel_classification/classification

RUN mkdir logs