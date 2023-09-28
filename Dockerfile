FROM --platform=linux/amd64 continuumio/miniconda3
RUN conda create -n env python=3.11
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ADD environment.yml environment.yml
RUN conda install conda=23.7.4
# RUN pip install gpt4all==1.0.12
RUN conda env create -f environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH


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