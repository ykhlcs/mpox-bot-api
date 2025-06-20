from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re
from sentence_transformers import SentenceTransformer, util

# ========================
# Shared Semantic Model
# ========================
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # Used for all similarity checks

# ========================
# Misinformation Prototypes
# ========================
misinfo_prototypes = [
    "Mpox is not spread by 5G transmission. It is spread through close physical contact and bodily fluids.",
    "Mpox can be cured instantly with garlic water.",
    "Mpox can spread through WiFi signals.",
    "5G towers release radiation that transmits mpox.",
    "Drinking garlic water prevents monkeypox infection.",
    "Garlic can protect you from catching mpox.",
    "Home remedies like garlic can prevent monkeypox."
]

def is_similar_to_misinformation(text, prototypes=misinfo_prototypes, threshold=0.75):
    emb_text = semantic_model.encode(text, convert_to_tensor=True)
    for prototype in prototypes:
        emb_proto = semantic_model.encode(prototype, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_text, emb_proto).item()
        if similarity > threshold:
            print(f"[DEBUG] Misinformation detected with similarity: {similarity:.3f}")
            return True
    return False

# ========================
# GPT-2 Perplexity Checker (lighter)
# ========================
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")

def get_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), 
            inputs.input_ids.view(-1)
        )
    return torch.exp(loss).item()

# ========================
# Reference Truths
# ========================
reference_statements = {
    "TRUE ✅": [
        "Mpox can spread through contaminated objects.",
        "Mpox transmission occurs via direct contact.",
        "Smallpox vaccines provide protection against mpox.",
        "Direct contact is the primary route for mpox transmission, though touching contaminated surfaces can also be a risk.",
        "Mpox spreads mainly through close, skin-to-skin contact."
    ],
    "FALSE ❌": [
        "Mpox is caused by 5G radiation.",
        "Mpox spreads through WiFi signals.",
        "Mpox can be cured instantly with garlic water.",
        "Drinking garlic water prevents monkeypox.",
        "Natural remedies can protect against mpox infection."
    ],
    "⚠️ UNCERTAIN": [
        "Some studies suggest that mpox might spread via contaminated surfaces, but the evidence remains inconclusive.",
        "There is mixed evidence on whether surface transmission plays a significant role in mpox spread."
    ]
}

label_urls = {
    "TRUE ✅": "https://www.who.int/news-room/questions-and-answers/item/mpox",
    "FALSE ❌": "https://www.who.int/news-room/fact-sheets/detail/monkeypox",
    "⚠️ UNCERTAIN": "https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html",
    "❓ Requires Expert Review": "https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html"
}

def is_nonsense(text: str) -> bool:
    return len(text.split()) < 2 or re.search(r"[^a-zA-Z0-9\s\.,!?]", text)

# ========================
# BERT Classification (Hugging Face Model)
# ========================
MODEL_DIR = "aerynnnn/mpox-mythbuster-bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

fact_checker = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)

# ========================
# Helper: Get explanation
# ========================
def get_dynamic_reason(user_text: str, label: str) -> str:
    candidate_reasons = {
        "TRUE ✅": [
            "Studies and trusted organizations confirm that direct contact is the main route of mpox transmission.",
            "Evidence shows mpox primarily spreads through close contact."
        ],
        "FALSE ❌": [
            "There is no credible scientific evidence supporting claims like 5G radiation or natural cures.",
            "No natural remedies prevent mpox. Only vaccines and avoiding exposure are proven prevention methods.",
            "Established health agencies explicitly warn against these unproven prevention claims."
        ],
        "⚠️ UNCERTAIN": [
            "Preliminary research has produced mixed results on this topic.",
            "The current evidence isn't strong enough to draw definitive conclusions."
        ],
        "❓ Requires Expert Review": [
            "The provided information requires expert analysis before reaching a conclusion.",
        ]
    }

    user_lower = user_text.lower()
    if "garlic" in user_lower and ("prevent" in user_lower or "protect" in user_lower):
        return "No natural remedies like garlic prevent mpox. The WHO explicitly warns against such unproven prevention methods."

    candidates = candidate_reasons.get(label, [])
    if not candidates:
        return "Additional details are unavailable."
    
    emb_query = semantic_model.encode(user_text, convert_to_tensor=True)
    best_candidate = max(
        candidates,
        key=lambda c: util.pytorch_cos_sim(emb_query, semantic_model.encode(c, convert_to_tensor=True)).item()
    )
    return best_candidate

# ========================
# Detect Misinformation
# ========================
def detect_misinformation(text):
    text_lower = text.lower()

    prevention_falsehoods = [
        "garlic water prevents", "garlic protects against", "home remedy prevents",
        "natural prevention", "herbal cure for", "garlic cure"
    ]
    if any(p in text_lower for p in prevention_falsehoods):
        return "Misinformation"

    if is_nonsense(text):
        return "Invalid Input"

    if "vaccine" in text_lower or "smallpox" in text_lower:
        return "Real"

    if is_similar_to_misinformation(text):
        return "Misinformation"

    results = fact_checker(text, top_k=None)
    flat_results = results[0] if isinstance(results[0], list) else results
    scores = {res['label'].lower(): res.get('score', 0) for res in flat_results}

    if scores.get("contradiction", 0) > scores.get("entailment", 0):
        return "Misinformation"
    elif scores.get("entailment", 0) > scores.get("contradiction", 0):
        return "Real"
    elif scores.get("neutral", 0) > 0.45:
        return "Uncertain"
    return "Requires Expert Review"

# ========================
# Final Classification Pipeline
# ========================
def classify_text(text: str):
    if not isinstance(text, str) or not text.strip():
        return ("Invalid Input", "⚠️ Sorry, I couldn't understand that.", "Input was empty.", None, 0.0)

    if is_nonsense(text):
        return ("Invalid Input", "⚠️ Gibberish detected.", "Input was not coherent.", None, 0.0)

    verdict = detect_misinformation(text)
    if verdict == "Misinformation":
        return "FALSE ❌", "This claim contradicts established scientific evidence.", get_dynamic_reason(text, "FALSE ❌"), label_urls["FALSE ❌"], 0.95
    elif verdict == "Real":
        return "TRUE ✅", "This statement aligns with verified health sources.", get_dynamic_reason(text, "TRUE ✅"), label_urls["TRUE ✅"], 0.95

    if "symptom" in text.lower() or "sign" in text.lower():
        return ("Informational", "Medical symptom inquiry", "This appears to be a request for symptom information", "https://www.cdc.gov/poxvirus/monkeypox/symptoms.html", 1.0)

    emb_text = semantic_model.encode(text, convert_to_tensor=True)
    avg_scores = {
        label: sum(util.pytorch_cos_sim(emb_text, semantic_model.encode(ref, convert_to_tensor=True)).item() for ref in refs) / len(refs)
        for label, refs in reference_statements.items()
    }

    best_label = max(avg_scores, key=avg_scores.get)
    highest_avg = avg_scores[best_label]
    SIMILARITY_THRESHOLD = 0.25
    if highest_avg < SIMILARITY_THRESHOLD:
        best_label = "❓ Requires Expert Review"

    explanations = {
        "TRUE ✅": "This statement aligns with verified health sources.",
        "FALSE ❌": "This claim contradicts established scientific evidence.",
        "⚠️ UNCERTAIN": "The evidence is limited or inconclusive regarding this claim.",
        "❓ Requires Expert Review": "Additional expert analysis is needed due to insufficient data."
    }

    return best_label, explanations[best_label], get_dynamic_reason(text, best_label), label_urls[best_label], highest_avg
