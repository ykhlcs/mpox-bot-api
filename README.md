# 🧠 Mpox Mythbuster API

This Flask API uses BERT and semantic similarity to classify whether a statement about Mpox (monkeypox) is:
- ✅ Real
- ❌ Misinformation
- ⚠️ Uncertain
- ❓ Requires Expert Review

### 🧪 How to Use

Send a `POST` request to `/classify` with JSON:

```bash
curl -X POST https://your-space-url.hf.space/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Garlic water cures mpox"}'
