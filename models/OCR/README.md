# Руководство по обучению OCR модели

Убедитесь, что у Вас установлен git lfs.

После этого необходимо скачать датасет:

```bash
git clone https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset
```

Установите зависимости с помощью команды uv sync (uv - пакетный мененджер, нужно установить).

Скачайте docker compose clearml с официального сайта, после чего запустите.

В терминале пропишите uv run clearml-init и вставьте credentials, которые можно создать в веб-панели clearml.

Запустите обучение, прописав 

```bash
    uv run python train_ocr.py --data_dir ./dataset \
                        --epochs 50 --batch_size 16 --lr 1e-4
```