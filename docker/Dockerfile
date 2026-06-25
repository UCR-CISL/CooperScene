FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build build-essential \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip openmim \
    && mim install \
        "mmengine>=0.10" \
        "mmcv>=2.0,<2.2" \
        "mmdet>=3.2,<3.4" \
        "mmdet3d>=1.4,<1.5" \
    && pip install --no-cache-dir \
        "numpy<1.25" spconv-cu120 einops tqdm pyyaml

ENV COOPERSCENE_ROOT=/workspace/CooperScene
WORKDIR /workspace/CooperScene

COPY requirements.txt ./
COPY models  ./models
COPY tools   ./tools

RUN python models/bevfusion/setup.py develop

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
