# =============================================================================
# GraspGen Thesis - Development Environment
# Unified: CUDA 12.6 + ROS2 Humble + SAM3 + GraspGen + Orbbec SDK
#
# Compatibility note:
#   SAM3  requires: Python 3.12, PyTorch 2.7+, CUDA 12.6+
#   GraspGen needs: Python 3.10+, PyTorch 2.1+ (works with 2.7)
#   → Unified on: Python 3.12, PyTorch 2.7, CUDA 12.6
#   → spconv-cu126 replaces spconv-cu120
#   → pointnet2_ops recompiled against CUDA 12.6
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# 1. ROS2 Humble
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
# 2. System Dependencies (build tools, Orbbec SDK deps, OpenGL)
# =============================================================================
RUN apt-get update && apt-get install -y \
    git wget unzip build-essential cmake ninja-build \
    python3-pip python3-dev python3.12 python3.12-venv python3.12-dev \
    libopencv-dev libusb-1.0-0-dev libudev-dev v4l-utils \
    libgflags-dev nlohmann-json3-dev libgoogle-glog-dev \
    libssl-dev libdw-dev mesa-utils libgl1 libgl1-mesa-glx \
    libegl1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# =============================================================================
# 3. PyTorch 2.7 + CUDA 12.6
# =============================================================================
RUN pip3 install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# =============================================================================
# 4. SAM3 - Clone and install from source
# =============================================================================
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && cd /opt/sam3 \
    && pip3 install --no-cache-dir -e ".[notebooks]"

# =============================================================================
# 5. GraspGen - Clone and install from source
# =============================================================================
RUN git clone https://github.com/NVlabs/GraspGen.git /opt/GraspGen \
    && cd /opt/GraspGen \
    && pip3 install --no-cache-dir -e .

# Install PointNet++ CUDA extensions (recompile for CUDA 12.6)
# Set compute capability for your GPU (8.6=RTX 3090, 8.9=RTX 4090, 9.0=H100)
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN cd /opt/GraspGen && bash install_pointnet.sh

# spconv for CUDA 12.6 (replaces spconv-cu120 from GraspGen requirements)
RUN pip3 install --no-cache-dir spconv-cu126

# PyTorch Geometric + extensions (for GraspGen graph networks)
RUN pip3 install --no-cache-dir \
    torch-geometric \
    torch-scatter \
    torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# =============================================================================
# 6. Common Python Dependencies
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# 7. Robotiq 3F Gripper
# =============================================================================
RUN pip3 install --no-cache-dir pymodbus pyserial

# =============================================================================
# 8. Download Model Weights (at build time, requires HF_TOKEN)
# =============================================================================
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN pip3 install --no-cache-dir huggingface_hub[cli] && \
    if [ -n "$HF_TOKEN" ]; then \
      echo "==> Downloading SAM3 weights from HuggingFace..." && \
      python3 -c " \
from huggingface_hub import snapshot_download; \
import os; \
token = os.environ.get('HF_TOKEN'); \
snapshot_download(repo_id='facebook/sam3', local_dir='/opt/models/sam3', token=token); \
print('SAM3 weights downloaded to /opt/models/sam3'); \
" && \
      echo "==> Downloading GraspGen model checkpoints..." && \
      python3 -c " \
from huggingface_hub import snapshot_download; \
import os; \
token = os.environ.get('HF_TOKEN'); \
snapshot_download(repo_id='adithyamurali/GraspGenModels', local_dir='/opt/models/graspgen', token=token); \
print('GraspGen weights downloaded to /opt/models/graspgen'); \
"; \
    else \
      echo "WARNING: HF_TOKEN not set. Model weights NOT downloaded."; \
      echo "Run scripts/download_models.sh inside the container after setting HF_TOKEN."; \
    fi

# =============================================================================
# 9. ROS2 Workspace - Clone Orbbec SDK ROS2
# =============================================================================
ENV ROS_WS=/ros2_ws
WORKDIR ${ROS_WS}

RUN mkdir -p src && cd src \
    && git clone --branch v2-main https://github.com/orbbec/OrbbecSDK_ROS2.git

# Install udev rules for Orbbec cameras
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
    echo "export PYTHONPATH=/opt/sam3:/opt/GraspGen:\${PYTHONPATH}" >> /root/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> /root/.bashrc

# =============================================================================
# 11. Entrypoint
# =============================================================================
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
