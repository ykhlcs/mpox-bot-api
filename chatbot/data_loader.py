import pandas as pd
import re
from src.utils.helpers import similarity
from sentence_transformers import SentenceTransformer, util
from src.scrapers.who_scraper import scrape_who_data
from datasets import load_dataset

# === Load all required splits from Hugging Face Dataset Hub ===
faq_df_csv = pd.DataFrame(load_dataset("aerynnnn/mpox-dataset", split="faq"))
followup_df = pd.DataFrame(load_dataset("aerynnnn/mpox-dataset", split="followup"))
who_df = pd.DataFrame(load_dataset("aerynnnn/mpox-dataset", split="who"))
cdc_df = pd.DataFrame(load_dataset("aerynnnn/mpox-dataset", split="cdc"))

# ===== HEALTH KEYWORDS =====
HEALTH_KEYWORDS = {
    "transmission": ["spread", "transmit", "catch", "infect", "get it"],
    "prevention": ["prevent", "avoid", "stop", "protection", "safe"],
    "symptoms": ["symptom", "sign", "feel", "experience"],
    "treatment": ["treat", "cure", "medicine", "vaccine"]
}

# ===== SCENARIO QUESTIONS =====
scenario_questions = [
    {"question": "can you get mpox from shaking hands", 
     "answer": "Risk from brief handshake is very low unless there's direct contact with lesions",
     "source": "CDC/WHO Guidelines"},
    {"question": "mpox transmission from surfaces", 
     "answer": "Possible but less common than direct contact. Virus survives 1–2 days on surfaces.",
     "source": "CDC/WHO Guidelines"},
    {"question": "is mpox airborne",
     "answer": "Not considered airborne like COVID-19. Requires prolonged face-to-face contact.",
     "source": "CDC/WHO Guidelines"},
    {"question": "can pets spread mpox",
     "answer": "Possible but rare. Isolate from pets if infected.",
     "source": "CDC/WHO Guidelines"}
]

# ===== QUERY EXPANSION =====
def expand_health_query(text):
    text = text.lower()
    for category, keywords in HEALTH_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return text + " " + " ".join(keywords)
    return text

# ===== TEXT CLEANING =====
def basic_clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip().lower()

# ===== FAQ CONSTRUCTION =====
try:
    scraped_faqs = scrape_who_data()
except Exception as e:
    print("⚠️ WHO scrape failed:", e)
    scraped_faqs = []

faq_df_scraped = pd.DataFrame(scraped_faqs)

if 'source' not in faq_df_scraped.columns:
    faq_df_scraped['source'] = 'WHO Scraped'

faq_df_scraped = faq_df_scraped.rename(columns={'Fact': 'question'})
faq_df_scraped['question'] = faq_df_scraped['question'].str.lower()

faq_df_csv['question'] = faq_df_csv['question'].str.lower()
faq_df_csv['answer'] = faq_df_csv['answer'].astype(str)

faq_df = pd.concat([faq_df_scraped, faq_df_csv], ignore_index=True)

scenario_df = pd.DataFrame(scenario_questions)
faq_df = pd.concat([faq_df, scenario_df], ignore_index=True)

faq_df = faq_df.drop_duplicates(subset=['question'])

# ===== EMBEDDING GENERATION =====
qa_model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = qa_model.encode(faq_df['question'].tolist(), convert_to_tensor=True)

# ===== FAQ MATCHING FUNCTION =====
def faq_match(user_input, threshold=0.65):
    expanded_input = expand_health_query(user_input.lower())
    input_embedding = qa_model.encode(expanded_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, faq_embeddings)[0]

    top_indices = cosine_scores.topk(3).indices
    top_scores = cosine_scores[top_indices].tolist()

    for idx, score in zip(top_indices, top_scores):
        if score >= threshold:
            answer = faq_df.iloc[idx.item()]['answer']
            if pd.isna(answer) or not isinstance(answer, str):
                continue
            return answer, score

    return None, max(top_scores) if top_scores else 0

# ===== TRAINING DATA (for classification model) =====
followup_df['source'] = 'Follow-Up'
followup_df['clean_text'] = followup_df['clean_text'].astype(str).apply(basic_clean)
followup_df['binary_class'] = followup_df['binary_class'].astype(int)
followup_df = followup_df[followup_df['binary_class'].isin([0, 1])]
followup_df = followup_df.drop_duplicates(subset=['clean_text'])

who_df['binary_class'] = 1
who_df['source'] = 'WHO'
who_df['clean_text'] = who_df['clean_text'].astype(str).apply(basic_clean)
who_df = who_df.drop_duplicates(subset=['clean_text'])

cdc_df['binary_class'] = 1
cdc_df['source'] = 'CDC'
cdc_df['clean_text'] = cdc_df['clean_text'].astype(str).apply(basic_clean)
cdc_df = cdc_df.drop_duplicates(subset=['clean_text'])

positive_df = pd.concat([who_df, cdc_df], ignore_index=True).drop_duplicates(subset=['clean_text'])

target_positive = 477
target_negative = 377

if len(positive_df) < target_positive:
    print(f"Warning: Only {len(positive_df)} positive samples available.")
    positive_sample = positive_df
else:
    positive_sample = positive_df.sample(n=target_positive, random_state=42)

negative_candidates = followup_df[followup_df['binary_class'] == 0].drop_duplicates(subset=['clean_text'])

if len(negative_candidates) < target_negative:
    print(f"Warning: Only {len(negative_candidates)} negative samples available.")
    negative_sample = negative_candidates
else:
    negative_sample = negative_candidates.sample(n=target_negative, random_state=42)

train_df = pd.concat([
    positive_sample[['clean_text', 'binary_class']],
    negative_sample[['clean_text', 'binary_class']]
], ignore_index=True)

train_df['clean_text'] = train_df['clean_text'].apply(basic_clean)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Final label distribution after cleaning and deduplication:\n", train_df['binary_class'].value_counts())

# ===== RULE-BASED & SOURCE CHECKING =====
def rule_based_check(text):
    text = text.lower()
    for _, row in train_df.iterrows():
        if row['clean_text'] in text:
            return "Real", 1.0
    return None

verified_sources = positive_df.copy()
verified_sources['clean_text'] = verified_sources['clean_text'].str.lower()

def source_check_override(user_input, threshold=0.85):
    user_input = user_input.lower()
    for _, row in verified_sources.iterrows():
        fact = row['clean_text']
        score = similarity(user_input, fact)
        if score >= threshold:
            return "Real", score
    return None
