# IMDB Sentiment Analysis API

> LSTM-модель классифицирует отзыв о фильме как positive/negative — прототип API для мониторинга тональности текста в реальном времени.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)]()
[![Test--Accuracy](https://img.shields.io/badge/Test%20Accuracy-83.5%25-yellow)]()
[![Overfit--gap](https://img.shields.io/badge/Train--Test%20gap-16pp-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Компании ежедневно получают тысячи отзывов и упоминаний в соцсетях — читать их вручную не масштабируется. Этот API оценивает текст на positive/negative за миллисекунды, без ручной разметки на инференсе.

---

## Структура проекта

```
ml_IMDB_DatasetOf_50K_MovieReviews/
├── .gitignore
├── readme.md
├── requirements.txt
└── IMDB_DatasetOf_50K_MovieReviews/
    ├── IMDB_DatasetOf_50K_MovieReviews.ipynb   # обучение, 30 эпох
    ├── main.py                                  # FastAPI inference service
    ├── model_Sentiment_..._MovieReviews.pth     # веса LSTM (state_dict)
    ├── vocab_Sentiment_..._MovieReviews.pth     # словарь, собран на train-сплите
    └── Text.txt
```

---

## Demo

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Absolutely loved this product, works perfectly!"}'
```

**Response** (реальный формат из `main.py`):
```json
{
  "prediction": 1,
  "sentiment": "positive",
  "message": "Отзыв позитивный",
  "probability_positive": 94.12,
  "probability_negative": 5.88
}
```

---

## Результаты

| Метрика | Значение |
|---|---|
| Train Accuracy (30 эпох) | **99.70%** |
| Test Accuracy | **83.54%** |
| Train/Test gap | **~16.2 п.п.** |

Ни F1, ни ROC-AUC, ни precision/recall по классам в ноутбуке не считались — только accuracy через прямой подсчёт `correct/total` на train и test dataloader'ах.

### Модель переобучена, и это видно по кривой loss

Обучение шло **30 эпох без early stopping и без LR-шедулера**. Loss стабильно падал весь цикл (534 → 18.5 к 30-й эпохе), при этом train accuracy дошла до 99.7% — модель практически выучила обучающую выборку наизусть, несмотря на `dropout=0.5`. 16-процентный разрыв train/test — прямое следствие: обучение стоило остановить раньше (по early stopping на val loss), а не доводить до 30-й эпохи.

**Честная рамка для резюме/собеса:** "обучил LSTM sentiment-классификатор на 50K отзывах, получил 83.5% test accuracy; заметил явное переобучение по train/test gap и знаю, как его лечить (early stopping, weight decay, меньше эпох)" — это сильнее, чем цифра 87% без объяснения происхождения.

---

## Датасет

- **Источник:** `torchtext.datasets.IMDB` — загружается программно, без ручного скачивания CSV
- **Объём:** 50,000 отзывов (25K train / 25K test)
- **Баланс классов:** 50/50
- **Словарь:** собран только на train-сплите — предотвращает утечку тестовых токенов в vocab

---

## Архитектура

```
POST /predict  {"text": "..."}
        │
        ▼
Tokenizer            ← basic_english
        │
        ▼
Vocab lookup         ← собран на train, <unk> для OOV
        │
        ▼
Embedding(vocab, 64)
        │
        ▼
LSTM(64 → 64)         ← hidden_dim=64, финальный hidden state h_n[-1]
        │
        ▼
Dropout(0.5)
        │
        ▼
Linear(64 → 2)
        │
        ▼
Softmax → argmax     ← label + вероятности
        │
        ▼
{"sentiment": "positive", "probability_positive": 94.12}
```

> `hidden_dim=64` (не 128) — подтверждено и архитектурой в ноутбуке (`SentimentModel(len(vocab))` использует дефолтный `hidden_dim=64`), и константой `HIDDEN_DIM = 64` в `main.py`.

**Ключевые решения:**

Vocab строится только на train — тест-токены без совпадений попадают в `<unk>`, что даёт честную out-of-sample оценку.

Модель и vocab грузятся один раз при старте приложения (`lifespan`), не на каждый запрос.

---

## Стек

| Слой | Технологии |
|---|---|
| Deep Learning | PyTorch 2.3, torchtext |
| API | FastAPI, Uvicorn, Pydantic |
| Данные | torchtext.datasets.IMDB, DataLoader |
| Сериализация | torch.save / torch.load (state_dict) |
| Среда | pip / venv, Colab-совместимо |

---

## API

| Метод | Эндпоинт | Вход | Выход |
|---|---|---|---|
| POST | `/predict` | `{"text": "string"}` | `{prediction, sentiment, message, probability_positive, probability_negative}` |

---

## Как запустить

```bash
git clone https://github.com/your-username/imdb-sentiment-api
cd imdb-sentiment-api
pip install -r requirements.txt
```

```bash
jupyter notebook IMDB_DatasetOf_50K_MovieReviews.ipynb
```

```bash
python IMDB_DatasetOf_50K_MovieReviews/main.py
# API → http://127.0.0.1:8000
# Swagger → http://127.0.0.1:8000/docs
```

---

## Next Steps

- [ ] Добавить early stopping по val loss — сейчас обучение не останавливается до 30-й эпохи вне зависимости от переобучения
- [ ] Посчитать F1/precision/recall по классам, а не только accuracy
- [ ] Сравнить с честным baseline (например, Logistic Regression на TF-IDF) — сейчас в проекте нет baseline вообще
- [ ] Weight decay или уменьшение эпох для сокращения train/test gap

---

[//]: # (## Автор)
[//]: # (**[Имя]** — [LinkedIn]&#40;https://linkedin.com/in/you&#41; | [GitHub]&#40;https://github.com/you&#41;)