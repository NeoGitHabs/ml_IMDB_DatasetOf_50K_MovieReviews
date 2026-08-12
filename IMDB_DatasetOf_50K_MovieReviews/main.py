import torch
import uvicorn
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from torchtext.data import get_tokenizer

BASE_DIR = Path(__file__).parent
HIDDEN_DIM = 64


# ── Model ──────────────────────────────────────────────────────────────────────
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=HIDDEN_DIM, output_dim=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    vocab = torch.load(
        BASE_DIR / "vocab_Sentiment_IMDB_DatasetOf_50K_MovieReviews.pth",
        map_location=device, weights_only=False,
    )
    vocab.set_default_index(vocab["<unk>"] if "<unk>" in vocab else 0)

    model = SentimentModel(len(vocab)).to(device)
    model.load_state_dict(torch.load(
        BASE_DIR / "model_Sentiment_IMDB_DatasetOf_50K_MovieReviews.pth",
        map_location=device,
    ))
    model.eval()

    app.state.vocab = vocab
    app.state.model = model
    yield


app = FastAPI(title="IMDB Sentiment Classifier", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text не должен быть пустым")
        return v


# ── Utils ──────────────────────────────────────────────────────────────────────
def preprocess(text: str, vocab) -> torch.Tensor:
    ids = [vocab[t] for t in tokenizer(text)]
    return torch.tensor([ids], dtype=torch.int64, device=device)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(item: TextIn):
    x = preprocess(item.text, app.state.vocab)

    with torch.no_grad():
        logits = app.state.model(x)
        proba  = F.softmax(logits, dim=1)[0].tolist()
        label  = int(torch.argmax(logits, dim=1).item())

    return {
        "prediction":            label,
        "sentiment":             "positive" if label == 1 else "negative",
        "message":               "Отзыв позитивный" if label == 1 else "Отзыв негативный",
        "probability_positive":  round(proba[1] * 100, 2),
        "probability_negative":  round(proba[0] * 100, 2),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)