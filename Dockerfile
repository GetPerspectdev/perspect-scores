# ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
# FROM nvidia/cuda:${CUDA_IMAGE}

# WORKDIR /app
# COPY ./requirements.txt .

# RUN apt-get update && apt-get upgrade -y \
#     && apt-get install -y git build-essential \
#     python3 python3-pip gcc wget \
#     ocl-icd-opencl-dev opencl-headers clinfo \
#     libclblast-dev libopenblas-dev \
#     && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# COPY . .

# # setting build related env vars
# ENV CUDA_DOCKER_ARCH=all
# ENV LLAMA_CUBLAS=1
# ENV PORT=80

# # Install depencencies
# RUN pip install --upgrade pip
# RUN pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings
# RUN pip install -r requirements.txt

# # Install llama-cpp-python (build with cuda)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

# COPY . .
# EXPOSE 80
# CMD ["python3", "web.py"]
FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "scores/web.py"]
