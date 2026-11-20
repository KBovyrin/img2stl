# Деплой img2stl (с нуля на машину с NVIDIA GPU)

Проверено на: Bazzite (Fedora Atomic), RTX 4080, драйвер 615.78.08, podman 5.8.9.
Подойдёт для любого Linux-хоста с podman ≥ 4.4 и nvidia-container-toolkit.

## 0. Требования

- **GPU NVIDIA**; образ собран под архитектуру **sm_89** (RTX 40xx). Для другой
  карты — см. раздел «Перенос на другое железо».
- **Драйвер** ≥ 530 (работает новая версия — обратная совместимость CUDA 12.1).
- **podman ≥ 4.4** и **nvidia-container-toolkit** (утилита `nvidia-ctk`).
- **Диск**: 25–30 ГБ (образ ~12 ГБ + веса ~6 ГБ + результаты).
- **Сеть** при первом запуске (HuggingFace, ~6 ГБ).

## 1. GPU-доступ в контейнеры (CDI)

Вариант A — с root (рекомендуется, работает для всех способов запуска):

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Вариант B — без root, в пользовательский каталог:

```bash
nvidia-ctk cdi generate --output=~/.config/cdi/nvidia.yaml
```

Для варианта B podman должен знать о каталоге со спекой —
`~/.config/containers/containers.conf`:

```ini
[containers]
cdi_spec_dir=["/var/home/<user>/.config/cdi"]
```

(наши `scripts/run.sh` / `scripts/dev.sh` передают `--cdi-spec-dir` сами;
`containers.conf` нужен только для systemd-автозапуска из раздела 4).

Важно: спека привязана к версии драйвера — **после каждого обновления драйвера
перегенерируйте её**, иначе контейнер потеряет GPU.

## 2. Сборка образа

```bash
cd pic2stl
podman build -t img2stl .        # 10–20 минут, все версии запинены в Containerfile
```

Самопроверка образа (импорты + CUDA) выполняется автоматически на сборке.

## 3. Первый запуск и проверка

```bash
scripts/run.sh                   # UI: http://127.0.0.1:7860
```

Первый Generate скачает веса (~6 ГБ, 10–15 мин) в volume `img2stl-cache`.
Проверка GPU и всего конвейера без UI:

```bash
scripts/dev.sh python -m img2stl.smoke --img samples/test.png --out out
```

В логе должно быть: `GPU: NVIDIA …`, время генерации ~15–20 с,
`watertight: False → True`, файлы `out/raw.glb`, `out/final.stl`.

## 4. Автозапуск (systemd + quadlet)

Остановите ручной контейнер (имя занято): `podman stop img2stl`.

Создайте `~/.config/containers/systemd/img2stl.container`:

```ini
[Unit]
Description=img2stl — фото → 3D-модель (Gradio UI)

[Container]
Image=localhost/img2stl:latest
ContainerName=img2stl
Pull=never
AddDevice=nvidia.com/gpu=all
SecurityLabelDisable=true
UserNS=keep-id
PublishPort=127.0.0.1:7860:7860
Volume=img2stl-cache:/cache
Volume=%h/pic2stl/out:/app/out

[Service]
Restart=on-failure

[Install]
WantedBy=default.target
```

Пути в `Volume=` поправьте под фактическое расположение проекта.

Включение:

```bash
systemctl --user daemon-reload
systemctl --user enable --now img2stl.service
loginctl enable-linger $USER     # работать без активного логина
```

Логи: `journalctl --user -u img2stl -f`.

## 5. Обновление приложения

```bash
# правки кода →
podman build -t img2stl .                 # быстро: кэш слоёв зависимостей
systemctl --user restart img2stl          # либо перезапуск run.sh вручную
```

Зависимости меняются только при правках `Containerfile` (тогда перескачаются
соответствующие слои; версии запинены — сборка воспроизводима).

## 6. Кэш весов: бэкап/перенос

```bash
podman volume export img2stl-cache -o img2stl-cache.tar     # ~6 ГБ
# на новой машине:
podman volume create img2stl-cache
podman volume import img2stl-cache img2stl-cache.tar
```

Альтернатива — просто дать весам скачаться заново при первом запуске.

## 7. Перенос на другое железо

- **uid пользователя** на хосте должен быть 1000 (в образе `USER app` c uid 1000
  + `--userns=keep-id`); иначе поправьте `useradd -u …` в `Containerfile`.
- **Другая GPU**: узнайте compute capability (`nvidia-smi --query-gpu=compute_cap`),
  замените в `Containerfile` `8.9` (TORCH_CUDA_ARCH_LIST) и `89`
  (CMAKE_CUDA_ARCHITECTURES) на нужные значения и пересоберите образ
  (слой torchmcubes пересоберётся).
- Переносится: каталог проекта + volume весов (раздел 6). Образ можно
  `podman save`/`load` или пересобрать на месте.

## 8. Диагностика

| Симптом | Действие |
|---|---|
| В контейнере нет GPU (`No CUDA GPUs`) | перегенерировать CDI-спеку (раздел 1) — типовая причина после обновления драйвера |
| Порт 7860 занят | `podman stop img2stl` или `PORT=7861 scripts/run.sh` |
| Логи приложения | `journalctl --user -u img2stl` или `podman logs img2stl` |
| Место закончилось | `podman system df`; чистка старых образов `podman image prune` |
| Первый запуск «завис» | идёт скачивание весов (~6 ГБ) — смотрите прогресс в логах |

## 9. Безопасность

UI без аутентификации (by design, локальный один пользователь). Публикация
только на `127.0.0.1:7860` — **не** меняйте на `0.0.0.0` без дополнительной
защиты (reverse-proxy с авторизацией и т.п.).
