ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
# FROM --platform=linux/amd64 ubuntu:rolling
ENV TZ=America

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN apt-get install bzip2
RUN apt-get install ca-certificates
RUN apt-get install libffi-dev
RUN apt-get install ncurses-dev
RUN apt-get install openssl
RUN apt-get install libreadline-dev
RUN apt-get install -y python-setuptools
RUN apt-get install -y sqlite
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y tk
RUN apt-get install -y python3-wheel
RUN apt-get install xz-utils
RUN apt-get install zlib1g

COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
ENV PORT=80
# build with cuda
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

# Install depencencies
RUN pip install --upgrade pip
RUN pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings gunicorn

RUN pip install -r requirements.txt

COPY . .
EXPOSE 80
CMD ["gunicorn", "--timeout", "0", "web:app"]







# FROM alpine:3.16

# RUN apk add --no-cache wget

# RUN apk add --update py-pip

# COPY requirements.txt requirements.txt
# RUN pip install gpt4all
# # RUN pip install numpy
# RUN pip install -r requirements.txt
# RUN pip install numpy

# COPY . .
# EXPOSE 5051
# CMD ["python3", "scores/test_app.py"]


# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# EXPOSE 5051
# # The code to run when container is started:
# ENTRYPOINT ["conda", "run", "-n", "myenv", "python3", "src/server.py"]

#############    #############    #############    #############    #############   #############
# ARG CUDA_IMAGE="12.1.1-devel-ubuntu22.04"
# FROM nvidia/cuda:${CUDA_IMAGE}

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
# ENV PORT=5051

# # Install depencencies
# RUN pip install --upgrade pip
# RUN pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings
# RUN pip install -r requirements.txt

# # Install llama-cpp-python (build with cuda)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

# COPY . .
# EXPOSE 5051
# CMD ["python3", "web.py"]