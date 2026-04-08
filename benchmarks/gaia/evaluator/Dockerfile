FROM python:3.11-slim

# Install common dependencies for GAIA tasks
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    requests \
    beautifulsoup4

WORKDIR /workspace

# This container will be used to execute agent-generated code
ENTRYPOINT ["python"]
