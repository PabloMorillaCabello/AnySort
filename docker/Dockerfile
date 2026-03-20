# =============================================================================
# GraspGen Thesis - Development Environment
# CUDA 12.6 + ROS2 Humble + GraspGen (uv venv) + SAM3 (Python 3.12 venv)
#
# Architecture:
#   - ROS2 Humble: Python 3.10 (system default, Ubuntu 22.04 debs)
#   - GraspGen: Python 3.10 (uv-managed venv at /opt/GraspGen/.venv)
#   - SAM3: Python 3.12 (isolated venv at /opt/sam3env)
#   - Dobot API: TCP/IP robot control (/opt/Dobot_hv, on PYTHONPATH)
#
# Compatibility:
#   - SAM3 requires: Python 3.12+, PyTorch 2.7+, CUDA 12.6+
#   - GraspGen works with: Python 3.10+, PyTorch (managed by uv)
#   - ROS2 Humble: Python 3.10 only (binary packages)
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# 1. ROS2 Humble (Python 3.10)
# =============================================================================
RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg lsb-release \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-moveit \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    ros-humble-controller-manager \
    ros-humble-joint-trajectory-controller \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-image-publisher \
    ros-humble-camera-info-manager \
    ros-humble-diagnostic-updater \
    ros-humble-diagnostic-msgs \
    ros-humble-statistics-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-xacro \
    ros-humble-backward-ros \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. System Dependencies (build tools, Python 3.12 for SAM3, OpenGL, etc.)
# =============================================================================
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt-get install -y \
    git wget unzip build-essential cmake ninja-build \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip python3-dev \
    libopencv-dev libusb-1.0-0-dev libudev-dev v4l-utils \
    libgflags-dev nlohmann-json3-dev libgoogle-glog-dev \
    libssl-dev libdw-dev mesa-utils libgl1 libgl1-mesa-glx \
    libegl1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3-tk python3.12-tk feh git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# =============================================================================
# 3. GraspGen - Install via uv with Python 3.10 venv
# =============================================================================
# Install uv package manager (handles dependency resolution much better than pip)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Limit parallel compilation jobs to prevent freezes on low-RAM systems
ENV MAX_JOBS=2

# Set compute capability for your GPU (8.6=RTX 3060, 8.9=RTX 4090, 9.0=H100)
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Clone GraspGen, create venv, install all deps + PointNet++ CUDA extensions
SHELL ["/bin/bash", "-c"]
RUN git clone https://github.com/NVlabs/GraspGen.git /opt/GraspGen \
    && cd /opt/GraspGen \
    && uv python install 3.10 \
    && uv venv --python 3.10 .venv \
    && source .venv/bin/activate \
    && uv pip install -e . \
    && chmod +x install_uv_pointnet.sh \
    && ./install_uv_pointnet.sh \
    && python tests/test_inference_installation.py

# Download GraspGen model checkpoints via git-lfs
RUN cd /opt/GraspGen \
    && git clone https://huggingface.co/adithyamurali/GraspGenModels

# =============================================================================
# 4. Additional pipeline dependencies (installed into GraspGen venv)
# =============================================================================
COPY docker/requirements.txt /tmp/requirements.txt
RUN source /opt/GraspGen/.venv/bin/activate \
    && uv pip install "numpy<2" \
    && uv pip install -r /tmp/requirements.txt \
    && uv pip install pymodbus pyserial pyorbbecsdk

# Make tkinter accessible in the venv (needed for Dobot UI).
# tkinter is a system C extension that can't be pip-installed; if the uv-managed
# Python doesn't bundle it, symlink the system copy into the venv's site-packages.
RUN /opt/GraspGen/.venv/bin/python -c "import tkinter" 2>/dev/null \
    || { SITE=$(/opt/GraspGen/.venv/bin/python -c "import site; print(site.getsitepackages()[0])") \
         && cp -a /usr/lib/python3.10/tkinter "$SITE/" 2>/dev/null \
         && cp -a /usr/lib/python3.10/lib-dynload/_tkinter*.so "$SITE/" 2>/dev/null \
         && echo "Symlinked system tkinter into GraspGen venv"; } \
    || echo "WARNING: tkinter not available (Dobot UI may not work)"

# =============================================================================
# 5. Dobot TCP/IP Robot API
# =============================================================================
# Python API for controlling Dobot robots via TCP/IP (ports 29999, 30004)
# Only needs numpy (already in GraspGen venv)
RUN git clone https://github.com/dauken85/Dobot_hv.git /opt/Dobot_hv

# Fix feed_back() TCP partial-read crash (ValueError: buffer size must be a
# multiple of element size). See docker/patches/fix_dobot_feedback.py
COPY docker/patches/fix_dobot_feedback.py /tmp/patches/fix_dobot_feedback.py
RUN python3 /tmp/patches/fix_dobot_feedback.py /opt/Dobot_hv/ui.py

# =============================================================================
# 6. SAM3 - Python 3.12 venv (separate from GraspGen due to Python version)
# =============================================================================
# Create isolated Python 3.12 environment for SAM3
RUN python3.12 -m venv /opt/sam3env

# Install PyTorch 2.7 + CUDA 12.6 in SAM3 venv
RUN /opt/sam3env/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/sam3env/bin/pip install --no-cache-dir \
       torch==2.7.0 \
       torchvision==0.22.0 \
       torchaudio==2.7.0 \
       --index-url https://download.pytorch.org/whl/cu126

# Install transformers (for HuggingFace SAM3 API)
RUN /opt/sam3env/bin/pip install --no-cache-dir setuptools transformers

# Clone SAM3 and install in the 3.12 venv
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && cd /opt/sam3 \
    && /opt/sam3env/bin/pip install --no-cache-dir -e ".[notebooks]"

# Download SAM3 weights (requires HF_TOKEN)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ -n "$HF_TOKEN" ]; then \
      echo "==> Downloading SAM3 weights from HuggingFace..." && \
      /opt/sam3env/bin/python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='facebook/sam3', local_dir='/opt/models/sam3', token=os.environ.get('HF_TOKEN')); print('SAM3 weights downloaded')" ; \
    else \
      echo "WARNING: HF_TOKEN not set. SAM3 weights NOT downloaded." ; \
      echo "Run scripts/download_models.sh inside the container after setting HF_TOKEN." ; \
    fi

# =============================================================================
# 7. Orbbec Camera - SDK + ROS2 Wrapper
# =============================================================================
# Install Orbbec SDK v2.7.6 (provides liborbbec, headers, and udev rules).
# The .deb is in data/ (tracked in repo). If not present, download from GitHub.
COPY data/OrbbecSDK_v2.7.6_amd64.deb /tmp/OrbbecSDK_v2.7.6_amd64.deb
RUN dpkg -i /tmp/OrbbecSDK_v2.7.6_amd64.deb; apt-get install -f -y \
    && rm /tmp/OrbbecSDK_v2.7.6_amd64.deb

# =============================================================================
# 8. ROS2 Workspace - Build OrbbecSDK_ROS2 + project packages
# =============================================================================
ENV ROS_WS=/ros2_ws
WORKDIR ${ROS_WS}

RUN mkdir -p src && cd src \
    && git clone --branch v2-main https://github.com/orbbec/OrbbecSDK_ROS2.git

# Install udev rules for Orbbec cameras (effective on host, baked into image)
RUN cd src/OrbbecSDK_ROS2/orbbec_camera/scripts \
    && chmod +x install_udev_rules.sh \
    && bash install_udev_rules.sh || true

# Initialize rosdep and install workspace dependencies
RUN rosdep init || true && rosdep update \
    && source /opt/ros/humble/setup.bash \
    && rosdep install --from-paths src --ignore-src -r -y || true

# Build the ROS2 workspace (OrbbecSDK_ROS2 + any packages in src/)
RUN source /opt/ros/humble/setup.bash \
    && colcon build --symlink-install \
       --cmake-args -DCMAKE_BUILD_TYPE=Release \
       --parallel-workers 2

# =============================================================================
# 9. Environment Setup
# =============================================================================
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Activate GraspGen Python 3.10 venv (uv-managed)" >> /root/.bashrc && \
    echo "source /opt/GraspGen/.venv/bin/activate" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/GraspGen:/opt/Dobot_hv:\${PYTHONPATH}" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Environment switching helpers" >> /root/.bashrc && \
    echo "alias sam3_activate='deactivate 2>/dev/null; source /opt/sam3env/bin/activate'" >> /root/.bashrc && \
    echo "alias graspgen_activate='deactivate 2>/dev/null; source /opt/GraspGen/.venv/bin/activate'" >> /root/.bashrc

# =============================================================================
# 10. Entrypoint
# =============================================================================
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
