import torch
from sentence_transformers import SentenceTransformer, util

SCENARIO_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Define evidence URLs FIRST
EVIDENCE_URLS = {
    "handshake": "https://www.cdc.gov/poxvirus/monkeypox/transmission.html",
    "surface": "https://www.who.int/news-room/questions-and-answers/item/monkeypox",
    "air": "https://www.cdc.gov/poxvirus/monkeypox/transmission.html#respiratory",
    "pool": "https://www.who.int/health-topics/monkeypox#tab=tab_3",
    "food": "https://www.cdc.gov/foodsafety/outbreaks/monkeypox.html"
}

SCENARIO_DB = {
    "handshake": {
        "label": "LOW RISK",
        "explanation": "Brief skin-to-skin contact poses minimal risk",
        "reason": "Requires direct contact with lesions or prolonged exposure",
        "evidence": "CDC states transmission requires direct contact with infectious rash"
    },
    "surface": {
        "label": "MODERATE RISK",
        "explanation": "Possible through contaminated objects",
        "reason": "Virus can survive on surfaces for limited time",
        "evidence": "WHO reports infection possible via fomites"
    },
    "air": {
        "label": "LOW RISK",
        "explanation": "Not considered airborne like COVID-19",
        "reason": "Requires prolonged face-to-face contact",
        "evidence": "CDC: Respiratory transmission only through prolonged exposure"
    },
    "pool": {
        "label": "VERY LOW RISK",
        "explanation": "Water transmission is highly unlikely",
        "reason": "Chlorine in pools kills the mpox virus",
        "evidence": "WHO: No documented cases of waterborne transmission"
    },
    "food": {
        "label": "LOW RISK",
        "explanation": "Food transmission is theoretically possible but rare",
        "reason": "Virus doesn't survive stomach acids well",
        "evidence": "CDC: No confirmed cases from food consumption"
    }
}

# Precompute embeddings
SCENARIO_EMBEDDINGS = {
    scenario: SCENARIO_MODEL.encode(scenario, convert_to_tensor=True)
    for scenario in SCENARIO_DB.keys()
}

def classify_scenario(text: str):
    text_lower = text.lower()
    emb_text = SCENARIO_MODEL.encode(text_lower, convert_to_tensor=True)
    
    best_match = None
    best_score = 0
    
    for scenario, data in SCENARIO_DB.items():
        # 1. Check for direct keyword match
        if scenario in text_lower:
            return (
                data["label"],
                data["explanation"],
                data["reason"],
                EVIDENCE_URLS.get(scenario, "https://www.cdc.gov/poxvirus/monkeypox/transmission.html"),
                0.95
            )
        
        # 2. Semantic similarity match
        score = util.pytorch_cos_sim(emb_text, SCENARIO_EMBEDDINGS[scenario]).item()
        if score > best_score:
            best_score = score
            best_match = (scenario, data)
    
    # 3. Confidence note determination
    if best_score > 0.8:
        confidence_note = "(High confidence assessment)"
    elif best_score > 0.65:
        confidence_note = "(Moderate confidence assessment)"
    else:
        confidence_note = "(Based on general guidelines)"
    
    # 4. Handle semantic match above threshold
    if best_match and best_score > 0.65:
        scenario_key, data = best_match
        return (
            data["label"],
            data["explanation"],
            f"{data['reason']} {confidence_note}",
            EVIDENCE_URLS.get(scenario_key, "https://www.cdc.gov/poxvirus/monkeypox/transmission.html"),
            best_score
        )
    
    # 5. Fallback to standard classification
    from .classifier import classify_text
    return classify_text(text)