# =============================================================================
# Cluster Dockerfile for triangle-splatting2-edit
#
# Before building, make sure submodules are initialized locally:
#   git submodule update --init --recursive
#
# Build:  docker build -f Dockerfile.cluster -t triangle-splatting:latest .
# Tag:    docker tag triangle-splatting:latest 195.251.117.42:31000/triangle-splatting:latest
# Push:   docker push 195.251.117.42:31000/triangle-splatting:latest
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
ARG CUDA_VERSION=13.0.1

# =============================================================================
# STAGE 1: Build Python Environment
# NOTE: Using devel (not runtime) — nvcc is required to compile CUDA extensions
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# =============================================================================
# SECTION 1: Regular Python Packages
# =============================================================================

# PyTorch 2.7.1 with CUDA 12.6 — installed before requirements.txt so
# packages that import torch at install time (mmcv, etc.) find it
RUN pip install --no-cache-dir torch==2.7.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# SECTION 2: Complex Packages (CUDA extensions, custom builds)
#
# TORCH_CUDA_ARCH_LIST uses semicolons, set INLINE (not via ENV)
# 9.0+PTX covers Blackwell (RTX 5090 / sm_120) via JIT compilation at first run
# =============================================================================

# Copy project code (submodules must be initialized locally before building)
WORKDIR /app
COPY . /app

# pytorch3d — pinned commit, requires CUDA arch list
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308"

# diff-triangle2-rasterization — local CUDA rasterizer submodule
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    pip install --no-cache-dir --no-build-isolation -e ./submodules/diff-triangle2-rasterization

# simple-knn — local submodule
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    pip install --no-cache-dir --no-build-isolation -e ./submodules/simple-knn

# xformers — pinned version matching PyTorch 2.7.1
RUN TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    pip install --no-cache-dir --no-build-isolation xformers==0.0.31

# triangulation — CMake/pybind11 build
RUN cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="/app/triangulation" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    -DPython3_EXECUTABLE="$(which python)" \
    && cmake --build build -j \
    && cmake --install build

# =============================================================================
# Verify Installation
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA {torch.version.cuda}')" || \
    (echo "ERROR: PyTorch import failed" && exit 1)
RUN python -c "import diff_triangle_rasterization; print('✓ diff_triangle_rasterization')"
RUN python -c "import simple_knn; print('✓ simple-knn')"

# =============================================================================
# STAGE 2: Runtime Image
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libtbb-dev \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Copy venv (compiled binaries) and app code from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# NOTE: NVIDIA_VISIBLE_DEVICES is set by the Kubernetes device plugin — do not hardcode
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYOPENGL_PLATFORM=egl
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN echo "===== Docker Image Build Summary =====" && \
    python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" && \
    echo "======================================"

CMD ["/opt/venv/bin/python", "train.py", "--help"]
