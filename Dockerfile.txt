FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential software-properties-common curl wget git unzip nano vim \
    python3 python3-pip python3-venv python3-dev ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements-audio.txt .

# Install Jupyter and dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-audio.txt && \
    pip install jupyter

COPY README.md .
COPY sample.ipynb .

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]