# img2stl — приложение целиком в контейнере (минимальное взаимодействие с хостом).
#
# Сборка:  podman build -t img2stl .
# Запуск:  scripts/run.sh          (UI: http://127.0.0.1:7860)
# Разработка/тесты: scripts/dev.sh pytest -q   (исходники монтируются поверх /app)
#
# С хостом контейнер взаимодействует только через:
#   - порт 127.0.0.1:7860
#   - GPU (CDI: --device nvidia.com/gpu=all)
#   - named volume img2stl-cache → /cache (веса HF + u2net)
#   - ./out → /app/out (результаты для скачивания)
#
# База: CUDA 12.1 toolkit под сборку torchmcubes; пайплайн работает на torch 2.5.1+cu121.

FROM docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

# Пины воспроизводимой сборки (см. также пины pip-пакетов ниже)
ARG HUNYUAN_COMMIT=f8db63096c8282cb27354314d896feba5ba6ff8a
ARG TORCHMCUBES_COMMIT=a9ac95aa329957ba646e7bb057f825f5dfa77f38

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    HF_HOME=/cache/hf \
    U2NET_HOME=/cache/u2net \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev git build-essential libgomp1 \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# 1) PyTorch cu121 (пин: у свежих torch нет cu121-сборок; для RTX 4080 = sm_89)
RUN pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# 2) hy3dgen (Hunyuan3D-2, только shape): пины чувствительных зависимостей,
#    сам пакет — с --no-deps, чтобы его непинленные requirements не разъехались
RUN pip install \
        "diffusers==0.32.2" "transformers==4.49.0" \
        "accelerate==1.15.0" "einops==0.8.2" "omegaconf==2.4.0" "tqdm==4.70.1" \
        "opencv-python-headless==5.0.0.93" "trimesh==5.2.0" "pymeshlab==2025.7.post1" \
        "pygltflib==1.16.5" "xatlas==0.0.11" \
        "ninja==1.13.2" "pybind11==3.1.0" "scikit-build-core==1.1.1" "cmake==4.4.4" \
    && git clone https://github.com/Tencent/Hunyuan3D-2.git /tmp/hy3dgen \
    && git -C /tmp/hy3dgen checkout "${HUNYUAN_COMMIT}" \
    && pip install --no-deps /tmp/hy3dgen \
    && rm -rf /tmp/hy3dgen

# 3) torchmcubes (marching cubes): сборка CUDA-extension против установленного torch.
#    GPU в контейнере на этапе сборки нет, поэтому архитектуру задаём явно (RTX 4080 = sm_89)
RUN SKBUILD_CONFIGURE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" \
    pip install --no-deps --no-build-isolation \
        "git+https://github.com/tatsy/torchmcubes.git@${TORCHMCUBES_COMMIT}"

# 4) зависимости приложения (все версии — пины: проверенный, верифицированный набор;
#    huggingface_hub <1.0: иначе конфликт с transformers 4.49;
#    onnxruntime нужен rembg, но не тянется им как зависимость автоматически;
#    lxml нужен trimesh для экспорта 3MF)
RUN pip install \
        "gradio==6.17.3" "trimesh==5.2.0" "pymeshfix==0.18.1" "rembg==2.0.69" \
        "onnxruntime==1.23.2" "pillow==12.3.0" "pytest==9.2.0" \
        "huggingface_hub==0.36.2" "lxml==6.1.3"

# Самопроверка образа: все критичные импорты на месте (включая тяжёлые модули hy3dgen)
RUN python -c "import torch, torchvision, hy3dgen, torchmcubes, diffusers, \
        transformers, trimesh, pymeshfix, rembg, gradio, cv2, numpy; \
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; \
        from hy3dgen.rembg import BackgroundRemover; \
        print('imports OK | torch', torch.__version__, '| cuda', torch.version.cuda)"

# Код приложения — последним слоем (меняется чаще всего)
WORKDIR /app
COPY img2stl/ img2stl/
COPY app.py .
COPY tests/ tests/

# Непривилегированный пользователь: uid 1000 совпадает с владельцем ./out на хосте
RUN useradd -u 1000 -m app \
    && mkdir -p /cache/hf /cache/u2net /app/out \
    && chown -R app:app /cache /app
USER app

EXPOSE 7860
CMD ["python", "app.py"]
