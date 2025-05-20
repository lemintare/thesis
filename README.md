# О проекте

### Тема диплома: Система детектирования проверки доступа к объекту на основе нейронных сетей посредством web-интерфейса

Переименуйте файл .env.example -> .env и задайте значения.

Обучите модели (инструкции в README соответствующих директорий) и положите их в папку website/backend/src/models (информация о структуре папки в README).

### Для Linux
Запустите проект следующей командой

```bash
docker compose up -d
```

### Для других ОС

Запустите postgres следующей командой

```bash
docker compose up postgres
```

Перейдите в папку website/backend и пропишите

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Перейдите в папку website/frontend и пропишите

```bash
npm run dev
```