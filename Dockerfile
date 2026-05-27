# =============================================================================
# Cluster Dockerfile for triangle-splatting2-edit
#
# Build:  docker build -t triangle-splatting-plus:latest .
# Tag:    docker tag triangle-splatting-plus:latest 195.251.117.42:31000/triangle-splatting-plus:latest
# Push:   docker push 195.251.117.42:31000/triangle-splatting-plus:latest
# =============================================================================

# =============================================================================
# STAGE 1: Build
# pytorch/pytorch base guarantees torch+CUDA versions are pre-matched (12.6)
# =============================================================================
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone repo and submodules
WORKDIR /app
RUN git clone https://github.com/mthodoris/triangle-splatting2-edit.git . && \
    git submodule update --init --recursive --remote

# Create venv inside the project, inherit nothing from conda base
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# Install torch into venv (same version/CUDA as base image — guaranteed to work)
RUN pip install --no-cache-dir "torch==2.7.1+cu126" \
    --index-url https://download.pytorch.org/whl/cu126

# Install regular dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Complex packages
# =============================================================================

RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308"

RUN pip install --no-cache-dir --no-build-isolation \
    -e ./submodules/diff-triangle2-rasterization

RUN pip install --no-cache-dir --no-build-isolation \
    -e ./submodules/simple-knn

RUN pip install --no-cache-dir --no-build-isolation xformers==0.0.31

# Pin CUDA_ARCHITECTURES on the triangulation target so torch's injected
# gencode flags (e.g. sm_50) don't trigger Eigen's sm_70 requirement
RUN echo 'set_target_properties(triangulation PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")' \
    >> /app/src/CMakeLists.txt

RUN cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="/app/triangulation" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    -DPython3_EXECUTABLE="$(which python)" \
    && cmake --build build -j \
    && cmake --install build

# =============================================================================
# Verify
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
RUN python -c "import diff_triangle_rasterization; print('✓ diff_triangle_rasterization')"
RUN python -c "import simple_knn; print('✓ simple-knn')"

# =============================================================================
# STAGE 2: Runtime
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYOPENGL_PLATFORM=egl
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

CMD ["/app/.venv/bin/python", "train.py", "--help"]
