# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    unzip \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ffmpeg \
    libssl-dev \
    libx11-6 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh && \
    $CONDA_DIR/bin/conda clean -ya

# Create conda env
RUN conda create -n p3dmm python=3.9 -y && \
    echo "conda activate p3dmm" >> ~/.bashrc

# Activate conda env for following RUN commands
SHELL ["conda", "run", "-n", "p3dmm", "/bin/bash", "-c"]

# Install torch + CUDA packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    conda install -y -c nvidia/label/cuda-11.8.0 \
        cuda-nvcc \
        cuda-cccl \
        cuda-cudart \
        cuda-cudart-dev \
        libcusparse \
        libcusparse-dev \
        libcublas \
        libcublas-dev \
        libcurand \
        libcurand-dev \
        libcusolver \
        libcusolver-dev

# Copy project files
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY pyproject.toml /workspace/pyproject.toml
COPY setup.py /workspace/setup.py


# Install python dependencies
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git@stable && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git && \
    pip install gdown && \
    pip install -r requirements.txt

COPY src/pixel3dmm/ /workspace/src/pixel3dmm/
RUN pip install -e .

# === PREPROCESSING SETUP ===
WORKDIR /workspace/src/pixel3dmm/preprocessing/
RUN rm -rf /workspace/src/pixel3dmm/preprocessing/facer \
    rm -rf /workspace/src/pixel3dmm/preprocessing/MICA \
    rm -rf /workspace/src/pixel3dmm/preprocessing/PIPNet
    
# Clone and setup facer
RUN git clone https://github.com/FacePerceiver/facer.git && \
    cd facer && \
    cp ../replacement_code/farl.py facer/face_parsing/farl.py && \
    cp ../replacement_code/facer_transform.py facer/transform.py && \
    pip install -e .

# Clone and setup MICA
RUN git clone https://github.com/Zielon/MICA && \
    cp replacement_code/mica_demo.py MICA/demo.py && \
    cp replacement_code/mica.py MICA/micalib/models/mica.py && \
    cd MICA && \
    conda env create -f environment.yml && \
    mkdir -p data && \
    gdown https://drive.google.com/drive/folders/1f-2HYWT3DUWrT5RMsaOmq6NvTYfDVhrm -O data/ --folder && \
    cd data/FLAME2020 && \
    unzip FLAME2020.zip -d ./ && \
    rm -rf FLAME2020.zip && \
    cd .. && \
    mkdir -p pretrained/ && \
    wget -O pretrained/mica.tar "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1" && \
    mkdir -p ~/.insightface/models/  && \
    wget -O ~/.insightface/models/antelopev2.zip "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"  && \
    unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2 && \
    wget -O ~/.insightface/models/buffalo_l.zip "https://keeper.mpdl.mpg.de/f/8faabd353cfc457fa5c5/?dl=1" && \
    unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l

# Clone and setup PIPNet
RUN git clone https://github.com/jhb86253817/PIPNet.git && \
    cd PIPNet/FaceBoxesV2/utils && chmod +x make.sh && ./make.sh && \
    cd ../../ && \
    mkdir -p snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/ && \
    gdown --id 1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C -O snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth

# Download pretrained weights
WORKDIR /workspace/pretrained_weights
RUN mkdir -p /workspace/pretrained_weights && \
    gdown --id 1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY -O ./uv.ckpt && \
    gdown --id 1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD -O ./normals.ckpt

# Final command
CMD ["bash"]