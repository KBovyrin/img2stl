# img2stl — фото → 3D-модель для печати

Локальное приложение: одна фотография → готовый к печати меш (STL/OBJ/3MF).
Всё исполняется в контейнере (podman), с хостом — только порт `127.0.0.1:7860`,
GPU (CDI) и папка `out/` с результатами. Веса моделей — в named volume.

**Документация:** [USAGE.md](USAGE.md) — пользование · [DEPLOY.md](DEPLOY.md) — установка/развёртывание с нуля

- Генерация: Hunyuan3D-2 shape (`tencent/Hunyuan3D-2`, fp16, CUDA) — ~16 с на RTX 4080
- Фон: rembg/u2net (кэшируется)
- Постобработка: чистка → ремонт (pymeshfix) → масштаб в мм → укладка на стол (min Z = 0)
- UI: Gradio (локальный, один пользователь)

## Использование

```bash
# сборка образа (~20 ГБ: база CUDA + torch + веса зависимостей)
podman build -t img2stl .

# запуск UI → http://127.0.0.1:7860
scripts/run.sh

# первый запуск скачает ~5–6 ГБ весов пайплайна + ~170 МБ u2net в volume img2stl-cache
```

В UI: загрузить фото → задать целевой размер (мм) и ось → выбрать форматы → **Generate**.
Файлы появляются в компоненте скачивания и в `./out/run_<время>/` на хосте.

## Разработка и тесты

```bash
scripts/dev.sh pytest -q                                   # юнит-тесты (этап 1, meshops)
scripts/dev.sh python -m img2stl.smoke --img samples/test.png --out out   # smoke инференса
```

`dev.sh` монтирует исходники поверх `/app` — пересборка образа не нужна.

## Структура

```
app.py                  Gradio UI
img2stl/preprocess.py   снятие фона (rembg) → RGBA + превью на белом
img2stl/inference.py    Hunyuan3D-2 shape: lazy singleton fp16/cuda, GPU-lock, OOM-ретраи
img2stl/meshops.py      валидация/ремонт/масштаб/экспорт (чистый Python, тестируемый)
img2stl/smoke.py        CLI smoke-прогон
tests/test_meshops.py   юнит-тесты постобработки
Containerfile           образ приложения (пины версий и коммитов внутри)
scripts/run.sh          запуск UI (прод)
scripts/dev.sh          тесты/разработка в контейнере
```

## Требования к хосту

podman + CDI-спека GPU: `nvidia-ctk cdi generate --output=~/.config/cdi/nvidia.yaml`
(на Bazzite системная `/etc/cdi/nvidia.yaml` бывает от старого драйвера — скрипты
используют пользовательскую спеку и `--security-opt label=disable`).

## Известные ограничения

- One-view генерация «додумывает» тыл объекта — геометрическая точность не гарантируется.
- Меш после нейросети чинится pymeshfix; если остался не-watertight — UI показывает предупреждение.
- Права на веса Hunyuan3D-2 — Tencent Community License (не MIT).

## Лицензия

Код проекта — [GLWTPL / Good Luck With That Public License](LICENSE).

Веса Hunyuan3D-2/2mv в репозиторий не входят: скачиваются при первом запуске
и распространяются по Tencent Community License (см. «Известные ограничения»).
