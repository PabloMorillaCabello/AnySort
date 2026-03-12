# =============================================================================
# GraspGen Thesis - Development Environment
# CUDA 12.6 + ROS2 Humble (Python 3.10) + SAM3 (Python 3.12 venv) + GraspGen
#
# Architecture:
#   - ROS2 Humble: Python 3.10 (system default, Ubuntu 22.04 debs)
#   - GraspGen: Python 3.10 (same as ROS2, shares PyTorch 2.7)
#   - SAM3: Python 3.12 (isolated venv at /opt/sam3env)
#
# Compatibility:
#   - SAM3 requires: Python 3.12+, PyTorch 2.7+, CUDA 12.6+
#   - GraspGen works with: Python 3.10+, PyTorch 2.1+ (tested with 2.7)
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
# 2. System Dependencies (build tools, Python 3.12, Orbbec SDK deps, OpenGL)
# =============================================================================
# Add deadsnakes PPA for Python 3.12 (needed for SAM3 venv only)
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt-get install -y \
    git wget unzip build-essential cmake ninja-build \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip python3-dev \
    libopencv-dev libusb-1.0-0-dev libudev-dev v4l-utils \
    libgflags-dev nlohmann-json3-dev libgoogle-glog-dev \
    libssl-dev libdw-dev mesa-utils libgl1 libgl1-mesa-glx \
    libegl1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3-tk python3.12-tk feh \
    && rm -rf /var/lib/apt/lists/*

# Upgrade system pip (for Python 3.10 - ROS2/GraspGen environment)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# 3. PyTorch 2.7 + CUDA 12.6 (for Python 3.10 - shared by ROS2 and GraspGen)
# =============================================================================
# Remove distutils-installed packages that conflict with PyTorch dependencies
# Remove distutils-installed system packages that conflict with pip upgrades
# (sympy conflicts with PyTorch, blinker with Flask/roboticstoolbox,
#  scipy compiled against NumPy 1.x breaks with NumPy 2.x)
RUN rm -rf /usr/lib/python3/dist-packages/sympy* \
           /usr/lib/python3/dist-packages/mpmath* \
           /usr/lib/python3/dist-packages/scipy* \
           /usr/lib/python3/dist-packages/blinker*

RUN pip3 install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126


# =============================================================================
# 4. GraspGen - Clone and install from source (Python 3.10)
# =============================================================================
# GraspGen pins torch==2.1.0 in pyproject.toml, but works fine with 2.7.
# Use --no-deps to skip its pinned torch, then install its other deps separately.
RUN git clone https://github.com/NVlabs/GraspGen.git /opt/GraspGen \
    && cd /opt/GraspGen \
    && pip3 install --no-cache-dir --no-deps -e . \
    && pip3 install --no-cache-dir \
       numpy==1.26.4 \
       hydra-core h5py webdataset scikit-learn scipy \
       tensorboard tensorboardx trimesh==4.5.3 diffusers==0.11.1 \
       addict yapf==0.40.1 \
       yourdfpy==0.0.56 imageio meshcat msgpack msgpack-numpy \
       timm==1.0.15 transformers tensordict huggingface-hub==0.25.2
# Note: pickle5 removed (built-in since Python 3.8)
# Note: sharedarray removed (often fails to compile, optional for GraspGen)
# Note: PyOpenGL will be installed by pyrender at its required version

# Limit parallel compilation jobs to prevent freezes on low-RAM systems
ENV MAX_JOBS=2

# Install PointNet++ CUDA extensions (recompile for CUDA 12.6)
# Set compute capability for your GPU (8.6=RTX 3060, 8.9=RTX 4090, 9.0=H100)
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# PointNet++ needs g++ as the CUDA host compiler
ENV CC=g++
ENV CXX=g++
RUN cd /opt/GraspGen \
    && pip3 install --no-cache-dir --no-build-isolation pointnet2_ops/

# Reset CC/CXX back to gcc after PointNet++ build
# (roboticstoolbox-python has C code using 'delete' as a function name,
#  which is a reserved keyword in C++ — so g++ would reject it)
ENV CC=gcc
ENV CXX=g++

# spconv for CUDA 12.6 (replaces spconv-cu120 from GraspGen requirements)
RUN pip3 install --no-cache-dir spconv-cu126

# PyTorch Geometric + extensions (for GraspGen graph networks)
# --no-build-isolation lets torch-scatter/torch-cluster see the installed PyTorch
# during compilation (otherwise pip creates a clean env without torch)
RUN pip3 install --no-cache-dir torch-geometric \
    && pip3 install --no-cache-dir --no-build-isolation torch-scatter torch-cluster

# =============================================================================
# 5. SAM3 - Create Python 3.12 venv and install SAM3 there
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

# Ensure setuptools is available (provides pkg_resources, required by sam3.model_builder)
# Also install transformers for the HuggingFace SAM3 API
RUN /opt/sam3env/bin/pip install --no-cache-dir setuptools transformers

# Clone SAM3 and install in the 3.12 venv
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && cd /opt/sam3 \
    && /opt/sam3env/bin/pip install --no-cache-dir -e ".[notebooks]"

# =============================================================================
# 6. Common Python Dependencies (Python 3.10 environment)
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
# Remove distutils-installed system packages that pip cannot uninstall cleanly
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# 7. Robotiq 3F Gripper (Python 3.10)
# =============================================================================
RUN pip3 install --no-cache-dir pymodbus pyserial

# =============================================================================
# 8. Download Model Weights (at build time, requires HF_TOKEN)
# =============================================================================
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download SAM3 weights using SAM3's Python 3.12 environment
RUN if [ -n "$HF_TOKEN" ]; then \
      echo "==> Downloading SAM3 weights from HuggingFace..." && \
      /opt/sam3env/bin/python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='facebook/sam3', local_dir='/opt/models/sam3', token=os.environ.get('HF_TOKEN')); print('SAM3 weights downloaded')" && \
      echo "==> Downloading GraspGen model checkpoints..." && \
      python3 -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='adithyamurali/GraspGenModels', local_dir='/opt/models/graspgen', token=os.environ.get('HF_TOKEN')); print('GraspGen weights downloaded')" ; \
    else \
      echo "WARNING: HF_TOKEN not set. Model weights NOT downloaded." ; \
      echo "Run scripts/download_models.sh inside the container after setting HF_TOKEN." ; \
    fi

# =============================================================================
# 9. ROS2 Workspace - Clone Orbbec SDK ROS2
# =============================================================================
ENV ROS_WS=/ros2_ws
WORKDIR ${ROS_WS}

# Use bash for sourcing ROS setup
SHELL ["/bin/bash", "-c"]

RUN mkdir -p src && cd src \
    && git clone --branch v2-main https://github.com/orbbec/OrbbecSDK_ROS2.git

# Install udev rules for Orbbec cameras (note: effective on host, not in container)
RUN cd src/OrbbecSDK_ROS2/orbbec_camera/scripts \
    && chmod +x install_udev_rules.sh \
    && bash install_udev_rules.sh || true

# Initialize rosdep and install workspace dependencies
RUN rosdep init || true && rosdep update \
    && source /opt/ros/humble/setup.bash \
    && rosdep install --from-paths src --ignore-src -r -y || true

# =============================================================================
# 10. Environment Setup
# =============================================================================
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/GraspGen:\${PYTHONPATH}" >> /root/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# SAM3 Python 3.12 environment activation helper" >> /root/.bashrc && \
    echo "alias sam3_activate='source /opt/sam3env/bin/activate'" >> /root/.bashrc && \
    echo "# Note: Run 'sam3_activate' to use SAM3 in Python 3.12 venv" >> /root/.bashrc

# =============================================================================
# 11. Entrypoint
# =============================================================================
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
