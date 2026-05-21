FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pytorch3d from source (pinned commit)
RUN MAX_JOBS=4 pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308"

COPY submodules/diff-triangle2-rasterization ./submodules/diff-triangle2-rasterization
RUN pip install --no-cache-dir -e ./submodules/diff-triangle2-rasterization

COPY submodules/simple-knn ./submodules/simple-knn
RUN pip install --no-cache-dir -e ./submodules/simple-knn

COPY . .

RUN cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="/workspace/triangulation" \
    -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
    -DPython3_EXECUTABLE="$(which python3)" \
    && cmake --build build -j \
    && cmake --install build

RUN pip install --no-cache-dir xformers==0.0.31

