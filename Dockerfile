FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="86;89;90;120"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

RUN python -m venv /app/.venv

RUN pip install --upgrade pip setuptools wheel packaging pybind11 ninja

RUN pip install --no-cache-dir torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/mthodoris/triangle-splatting2-edit.git

WORKDIR /app/triangle-splatting2-edit

RUN git submodule update --init --recursive --remote

RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308"

RUN pip install --no-cache-dir --no-build-isolation -e ./submodules/diff-triangle2-rasterization

#RUN pip install --no-cache-dir --no-build-isolation -e ./submodules/simple-knn
ENV PYTHONPATH="/app/triangle-splatting2-edit:/app/triangle-splatting2-edit/submodules/simple-knn:${PYTHONPATH}"

RUN cd ./submodules/simple-knn && \
    pip install --no-cache-dir --no-build-isolation -e .

RUN cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="/app/triangle-splatting2-edit/triangulation" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    -DPython3_EXECUTABLE="$(which python)" \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;90;120" \
    && cmake --build build -j \
    && cmake --install build

RUN pip install --no-cache-dir --no-build-isolation xformers==0.0.31.post1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
RUN python -c "import diff_triangle_rasterization; print('diff_triangle_rasterization ok')"
RUN python -c "import simple_knn._C; print('simple_knn ok')"


ENTRYPOINT ["bash", "-c", "git pull && exec \"$@\"", "--"]
CMD ["python", "train.py", "--help"]