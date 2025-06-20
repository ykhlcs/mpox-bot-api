import os
import csv
import uuid
import logging
import random
from datetime import datetime, timedelta
import math
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters, ConversationHandler
)
import re
from transformers import pipeline

# Relative imports
from chatbot.classifier import classify_text
from chatbot.classifier_scenario import classify_scenario
from chatbot.data_loader import rule_based_check, faq_match, source_check_override
from chatbot.database import (
    init_db,
    log_user,
    log_message,
    log_misinformation,
    log_response
)
from chatbot.fetch_mpox_news import fetch_monkeypox_news

# ===== Hugging Face Spaces Configuration =====
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]  # Get token from HF secrets

# ADD HF-specific logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# States for conversation flow
CLARIFY, FOLLOW_UP = range(2)

# User context storage
USER_CONTEXT = {}

# Initialize the summarizer pipeline with t5-small model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants and cache
FEEDBACK_LOG_FILE = "feedback_log.csv"

RESPONSES = {
    "greeting": [
        "👋 Hi there! How can I help you today?",
        "Hello! Feel free to send me monkeypox news or ask questions.",
        "Hey! Ready to bust some monkeypox myths together!",
        "Hi! Send me any news headline and I'll check it for you."
    ],
    "casual_reply": [
        "😄 You're welcome!",
        "👍 Gotcha!",
        "✨ Cool cool!"
    ],
    "fallback": [
        "😕 Sorry, I couldn't understand that. Could you rephrase it?",
        "🤔 I'm not sure how to respond. Try sending me a news headline or question about monkeypox.",
        "🙈 That doesn't seem related to monkeypox. Want to try again?"
    ],
    "joke": [
        "🧪 Why don't viruses tell jokes? They don't have a good sense of humor - they prefer a good host instead!",
        "🦠 What did one cell say to his sister cell who stepped on his foot? Mitosis!",
        "🔬 Why did the bacteria cross the microscope? To get to the other slide!",
        "🧬 Why did the DNA go to therapy? It had too many unresolved pairs!",
        "🧴 Why did the hand sanitizer break up with the soap? It felt their relationship was too superficial!",
        "💉 What do you call a doctor who fixes websites? A URLologist!",
        "🩺 Why don't scientists trust atoms? Because they make up everything!",
        "🦠 What's a virus's favorite game? Hide and seek - because they're always in your cells!",
        "🧫 Why did the microbiologist become a gardener? They wanted to study the root of all pathogens!",
        "🦠 How do viruses communicate? Through the web!",
        "🩹 Why did the bandage go to the party? It was a wrap!",
        "🧼 What did the soap say to the germ? You're washed up!",
        "🔭 Why did the microbiologist bring a ladder to the bar? They heard the drinks were on the house!"
    ]
}

# ===== NEWS DETECTION ADDITION =====
NEWS_KEYWORDS = ["news", "update", "headline", "recent", "latest", "new", "current"]
# ===================================

CASUAL_KEYWORDS = [
    "thank you", "thanks", "thx", "thanx", "thank", 
    "appreciate", "cheers", "grateful", "kudos",
    "hank you", "hanks", "ty", "tq", "thnks", "thanku",
    "cool", "ok", "fine", "alright", "got it", "awesome"
]

GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", 
             "how are you", "how's it going", "what's up", "how do you do", "how are things",
             "how have you been", "how is everything"]

QUESTION_KEYWORDS = ['what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'can', 'does', 'do', 'will']

RISK_KEYWORDS = ["dangerous", "safe", "risk", "risky", "exposure", "contagious", "infect", "near", "close", "proximity"]

QUESTION_PHRASES = ["tell me", "explain", "describe", "list", "what is", "what are", "how do", "can you"]

# Add to priority handling after transmission
TRANSMISSION_SCENARIOS = [
    "handshake", "hand shake", "shake hands", "hug", "kiss", "embrace",
    "surface", "object", "toilet", "bedding", "clothing", "utensil",
    "air", "breathe", "cough", "sneeze", "respiratory", "aerosol",
    "water", "pool", "swim", "swimming", "beach", "ocean",
    "food", "eat", "drink", "animal", "pet"
]

MISINFO_KEYPHRASES = [
    "garlic water", "5g", "government hoax", "bill gates", "microchip",
    "wifi signals", "not real", "fake virus", "planned", "bio weapon"
]

# Off-topic detection
OFF_TOPIC_KEYWORDS = {
    "capital", "president", "weather", "sports", "movie", "music",
    "celebrity", "recipe", "game", "sport", "team", "actor", "actress",
    "book", "song", "artist", "football", "basketball", "entertainment",
    "history", "geography", "politics", "economy", "stock", "finance",
    "recipe", "cook", "food", "restaurant", "travel", "destination",
    "language", "translate", "currency", "population", "size"
}

JOKE_TRIGGERS = {
    "joke", "funny", "humor", "laugh", "hilarious", "comedy", 
    "kidding", "jest", "gag", "pun", "rofl", "lol", "make me laugh",
    "cheer me up", "tell me something funny", "lighten up"
}

# ===== CONTEXT MANAGEMENT =====
def get_user_context(user_id):
    """Get user context with expiration check"""
    clear_expired_context()
    return USER_CONTEXT.get(user_id, None)

def update_user_context(user_id, query, response_type, content):
    """Standardized context storage"""
    USER_CONTEXT[user_id] = {
        "type": response_type,
        "query": query,
        "content": content,
        "timestamp": datetime.now()
    }

def clear_expired_context():
    """Remove contexts older than 5 minutes"""
    now = datetime.now()
    expired_users = []
    
    for user_id, context in USER_CONTEXT.items():
        if (now - context['timestamp']) > timedelta(minutes=5):
            expired_users.append(user_id)
            
    for user_id in expired_users:
        del USER_CONTEXT[user_id]

# ===== Vague Reference Detection =====
def is_vague_reference(text: str) -> bool:
    """Check if message is a vague reference"""
    if not text.strip():
        return False
        
    text_lower = text.lower()
    vague_terms = ["this", "that", "it", "explain", "more", "detail", "tell me"]
    
    # Special cases for standalone phrases
    if text_lower in ["tell me", "explain", "what about", "how about", "and"]:
        return True
        
    # Check for short phrases with vague terms
    return (
        any(term in text_lower for term in vague_terms) 
        and len(text.split()) <= 3
    )

def is_clear_misinfo(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in MISINFO_KEYPHRASES)

def is_transmission_scenario(text: str) -> bool:
    text_lower = text.lower()
    
    # Enhanced scenario detection
    scenario_patterns = [
        r"can you get \w+ from",
        r"is it safe to",
        r"risk of.*from",
        r"transmi(t|ssion).*(through|via|from)",
        r"spread.*(in|through|at)"
    ]
    
    return (
        any(scenario in text_lower for scenario in TRANSMISSION_SCENARIOS) or
        any(re.search(pattern, text_lower) for pattern in scenario_patterns)
    )

# ===== OFF-TOPIC & JOKE HANDLING =====
def is_off_topic(text: str) -> bool:
    text_lower = text.lower()
    
    # 1. Check for explicit off-topic keywords
    if any(kw in text_lower for kw in OFF_TOPIC_KEYWORDS):
        return True
        
    # 2. Allow disease comparisons
    disease_keywords = {"pox", "virus", "disease", "illness", "infection"}
    if any(kw in text_lower for kw in disease_keywords):
        return False
        
    # 3. Check for absence of health-related keywords
    health_keywords = {"health", "medical", "clinic", "doctor", "hospital", "patient"}
    if not any(kw in text_lower for kw in health_keywords | {"monkeypox", "mpox"}):
        return classify_off_topic(text_lower)
    
    return False

def classify_off_topic(text: str) -> bool:
    """Use pattern matching to detect off-topic queries"""
    patterns = [
        r"who (is|are) .+",
        r"what (is|are) .+",
        r"where is .+",
        r"when (was|did) .+",
        r"how to .+",
        r"capital of",
        r"president of",
        r"leader of",
        r"population of",
        r"define ",
        r"meaning of",
        r"translate "
    ]
    return any(re.search(pattern, text) for pattern in patterns)

def is_joke_request(text: str) -> bool:
    """Detect requests for jokes"""
    text_lower = text.lower()
    # Check for joke keywords
    if any(trigger in text_lower for trigger in JOKE_TRIGGERS):
        return True
        
    # Check specific patterns
    patterns = [
        r"tell me a joke",
        r"make me laugh",
        r"cheer me up",
        r"lighten up",
        r"say something funny"
    ]
    return any(re.search(pattern, text_lower) for pattern in patterns)

# Helpers

def is_risk_query(text: str) -> bool:
    text_lower = text.lower()
    risk_phrases = [
        "safe to", "is it safe", "how safe", "should i worry", 
        "chance of getting", "likely to catch", "risk of",
        "more dangerous", "less dangerous", "compared to",  # ADDED
        "versus", "vs "  # ADDED
    ]
    return any(phrase in text_lower for phrase in risk_phrases)

def is_greeting(text):
    if not text.strip():
        return False
        
    text_lower = text.lower()
    for greet in GREETINGS:
        if re.search(rf"\b{re.escape(greet)}\b", text_lower):
            return True
    return False

def random_response(category):
    return random.choice(RESPONSES[category])

# ===== NEWS DETECTION =====
def is_news_request(text: str) -> bool:
    text_lower = text.lower()
    news_keyword_present = any(kw in text_lower for kw in NEWS_KEYWORDS)
    
    # Always allow news requests without disease keywords in this bot's context
    if news_keyword_present:
        return True
        
    # Keep optional disease keyword matching
    return ("mpox" in text_lower or "monkeypox" in text_lower) and any(
        kw in text_lower for kw in ["update", "report", "headline", "show", "give"]
    )

def is_casual_thanks(text):
    text_lower = text.lower()
    # Convert to set for faster lookups
    casual_set = {
        "thanks", "thank", "thx", "tx", "appreciate", 
        "ok", "cool", "lame", "fine", "alright", "got it",
        "cheers", "kudos", "ty"
    }
    
    # Check for any keyword match
    if any(k in text_lower for k in casual_set):
        return True
        
    # Handle common misspellings
    misspellings = {"hank", "tank", "thnak", "thnks", "thx"}
    return any(m in text_lower for m in misspellings)

def is_general_question(text: str) -> bool:
    text_lower = text.lower().strip()
    
    # Check for question mark
    if text_lower.endswith('?'):
        return True
        
    # Check first word
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in QUESTION_KEYWORDS:
        return True
        
    # Check for question phrases anywhere
    if any(phrase in text_lower for phrase in QUESTION_PHRASES):
        return True
        
    return False

async def handle_vague_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_context = get_user_context(user_id)
    
    # Check if we have previous context
    if user_context:
        response = (
            f"❓ Are you asking about:\n"
            f"\"{user_context['query']}\"?\n\n"
            "If yes, reply 'yes'. If not, please rephrase your question."
        )
    else:
        response = (
            "🤔 I'm not sure what you're referring to.\n"
            "Please provide more context or ask a complete question about monkeypox."
        )
    
    await update.message.reply_text(response)
    return CLARIFY

async def handle_clarification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text.lower()
    user_context = get_user_context(user_id)
    
    if user_context and ("yes" in user_text or "yeah" in user_text or "yep" in user_text):
        # Provide detailed explanation of previous response
        await explain_in_detail(update, user_context)
        return ConversationHandler.END
    else:
        await update.message.reply_text("👍 Got it. Please ask your new question clearly.")
        return ConversationHandler.END

async def explain_in_detail(update: Update, context_data):
    """Provide detailed explanation based on context type"""
    response_type = context_data["type"]
    content = context_data["content"]
    
    if response_type == "classification":
        response = (
            f"🔍 *Detailed Explanation:*\n\n"
            f"**Original Query:** {context_data['query']}\n"
            f"**Classification:** {content['label']}\n"
            f"**Reason:** {content['reason']}\n"
            f"**Full Explanation:** {content['explanation']}\n\n"
            f"📚 *Trusted Source:*\n{content['source_url']}"
        )
    
    elif response_type == "faq":
        response = (
            f"📖 *Full Answer:*\n\n"
            f"{content['answer']}\n\n"
            f"💡 *Summary:* {content['summary']}"
        )
    
    elif response_type == "info":
        category = content.get("category", "information")
        response = (
            f"📚 *Detailed {category.capitalize()} Information:*\n\n"
            "For comprehensive guidelines, please visit:\n"
            "🔗 [CDC Mpox Resources](https://www.cdc.gov/poxvirus/monkeypox)\n"
            "🔗 [WHO Mpox Q&A](https://www.who.int/news-room/questions-and-answers/item/mpox)"
        )
    
    elif response_type == "news":
        response = (
            f"📰 *News Details:*\n\n"
            f"**Headline:** {content['title']}\n"
            f"**Source:** {content['url']}\n\n"
            "ℹ️ For more news updates, visit trusted health news sources."
        )
    
    else:
        response = "ℹ️ Here's more information:\nhttps://www.cdc.gov/poxvirus/monkeypox"
    
    await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)

async def send_typing(context: ContextTypes.DEFAULT_TYPE, chat_id):
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Conversation cancelled. How else can I help you?")
    return ConversationHandler.END

# Updated function
def get_short_answer(text):
    """Safely summarize text handling all input types"""
    # Handle null/empty values
    if text is None or not isinstance(text, str) or not text.strip():
        return "No summary available"
    
    # Handle NaN values specifically
    if isinstance(text, float) and math.isnan(text):
        return "Information unavailable"
    
    # Only summarize long text
    word_count = len(text.split())
    if word_count > 100:
        try:
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:300] + "..." if len(text) > 300 else text
    return text

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(context, update.effective_chat.id)
    await update.message.reply_text(
        "👋 *Hi there!* I'm your personal Monkeypox News Verifier.\n\n"
        "📰 Just send me a news headline or short paragraph, and I'll check if it sounds real or suspicious.\n"
        "✅ Let's stop the spread of misinformation — together!\n"
        "Start by sending /start or /help for more info about me.",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(context, update.effective_chat.id)
    await update.message.reply_text(
        "🆘 *How to use this bot:*\n"
        "• 📝 *Send a message* — Paste any monkeypox-related news or claim.\n"
        "• 🧠 *Get instant analysis* — I'll show how likely it's real or fake.\n"
        "• 👍👎 *Give feedback* — Help improve future predictions.\n"
        "• ⚠️ *Report issues* — Found something wrong? Let me know.\n"
        "• 🔄 Use /start anytime to restart.\n\n"
        "Let's stay informed and safe! 💬",
        parse_mode="Markdown"
    )

# Classification logic
def post_process_verdict(claim_text, model_label, confidence):
    health_keywords = ['monkeypox', 'covid', 'vaccine', 'virus', 'pandemic']
    is_health_topic = any(keyword in claim_text.lower() for keyword in health_keywords)

    confidence_percent = confidence * 100

    if is_health_topic:
        if confidence_percent < 80:
            return 'Misinformation'
        elif 80 <= confidence_percent < 90:
            return 'Requires Expert Review'
        else:
            return model_label
    else:
        return 'Misinformation' if confidence_percent < 70 else model_label

def normalize_query(query: str) -> str:
    # Normalize by converting to lowercase and replacing synonyms
    normalized = query.lower().replace("monkeypox", "mpox")
    return normalized

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text.strip()
    user_id = str(user.id)
    user_text = normalize_query(message_text)
    lower_text = user_text.lower()
    confidence = None
    response_text = "No response generated"
    
    # Clear expired contexts at start
    clear_expired_context()
    
    # ===== PRIORITY 0: Joke Requests =====
    if is_joke_request(user_text):
        await update.message.reply_text(
            f"🦠 Here's a health-related joke for you:\n\n"
            f"{random_response('joke')}\n\n"
            f"😄 Now, how can I help with monkeypox information today?"
        )
        return
    
    # ===== PRIORITY 1: Off-Topic Queries =====
    if is_off_topic(user_text):
        responses = [
            "🤖 I'm specialized in monkeypox health information. I can help with:\n"
            "- Monkeypox symptoms and prevention\n"
            "- News verification\n"
            "- Transmission risks\n\n"
            "Try asking about these health topics instead!",
            
            "🔍 That seems outside my expertise. I focus specifically on monkeypox "
            "health information. Need help with symptoms or latest news?",
            
            "⚠️ I'm designed for monkeypox health information. For general knowledge, "
            "you might want to try a different service. I can help with:\n"
            "- Fact-checking monkeypox claims\n"
            "- Understanding transmission risks\n"
            "- Latest monkeypox news"
        ]
        await update.message.reply_text(random.choice(responses))
        return
    
    # ===== PRIORITY 2: Vague References =====
    if is_vague_reference(user_text):
        response_text = "Requesting clarification"
        return await handle_vague_query(update, context)
    
    # Add default response for unhandled cases
    if response_text == "No response generated":
        response_text = random_response("fallback")
        await update.message.reply_text(response_text)
    
    # ===== PRIORITY 3: Clear Misinformation =====
    if is_clear_misinfo(user_text):
        label, explanation, reason, url, _ = classify_text(user_text)
        response = (
            f"🤖 Prediction: *{label}*\n"
            f"📖 Explanation: {explanation}\n"
            f"📝 Reason: {reason}"
        )
        if url:
            response += f"\n🔗 [Source]({url})"
        await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context properly
        update_user_context(user_id, user_text, "classification", {
            "label": label,
            "explanation": explanation,
            "reason": reason,
            "source_url": url
        })
        return

    # ===== PRIORITY 4: Transmission Explanation =====
    if re.search(r"how is mpox (transmitted|spread)", lower_text):
        response_text = (
            "Mpox is primarily transmitted through prolonged, close, direct contact with an infected person – "
            "especially via skin-to-skin contact. Although transmission through contaminated surfaces is possible, "
            "it is considerably less common. For more details, please refer to the [CDC Mpox FAQ](https://www.cdc.gov/mpox/index.html) "
            "or [WHO Mpox Q&A](https://www.who.int/news-room/questions-and-answers/item/mpox)."
        )
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "info", {
            "category": "transmission"
        })
        return

    # ===== PRIORITY 5: Risk/Safety Queries =====
    if is_risk_query(user_text):
        # Special handling for disease comparisons
        if " vs " in user_text or "compared to" in user_text:
            diseases = ["covid", "chickenpox", "smallpox", "measles", "flu"]
            if any(disease in user_text for disease in diseases):
                response_text = (
                    "🔍 *Disease Comparison:*\n\n"
                    "Mpox vs other diseases:\n"
                    "• Fatality rate: 1-10% (lower than smallpox)\n"
                    "• Contagiousness: Less than measles or COVID\n"
                    "• Severity: Generally milder than smallpox\n\n"
                    "✅ *Trusted Comparison:*\n"
                    "🔗 [WHO Mpox vs Smallpox](https://www.who.int/news-room/questions-and-answers/item/monkeypox)\n"
                    "🔗 [CDC Mpox vs Chickenpox](https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html)"
                )
                await update.message.reply_text(response_text, parse_mode="Markdown")
                return

    # ===== PRIORITY 6: Symptom Queries =====
    if "symptom" in lower_text or "sign" in lower_text:
        faq_answer, faq_score = faq_match(user_text)
        if faq_answer:
            summary = get_short_answer(faq_answer)
            response_text = (
                "📘 *Informational Answer:*\n\n"
                f"_{summary}_\n\n"
                "✅ *For more details, check:* \n"
                "🔗 [CDC Mpox FAQ](https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html) | "
                "🔗 [WHO Mpox Overview](https://www.who.int/health-topics/monkeypox)"
            )
        else:
            response_text = (
                "🤔 *Hmm... I couldn't find an exact answer for that.*\n\n"
                "🔍 *But it's always best to refer to trusted health sources!* \n"
                "✅ [CDC Mpox FAQ](https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html) | "
                "[WHO Mpox Overview](https://www.who.int/health-topics/monkeypox)"
            )
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "faq", {
            "question": user_text,
            "answer": faq_answer,
            "summary": summary
        })
        return

    # ===== PRIORITY 7: Transmission Claims =====
    if any(kw in lower_text for kw in ["spread", "transmit", "catch", "infect", "exposure", "contact"]):
        faq_answer, faq_score = faq_match(user_text, threshold=0.6)
        
        # Handle cases where no FAQ match was found
        if faq_answer and not pd.isna(faq_answer):
            summary = get_short_answer(faq_answer)
            response_text = (
                "🔄 *Transmission Facts:*\n\n"
                f"_{summary}_\n\n"
                "✅ *Trusted Sources:*\n"
                "🔗 [CDC Transmission](https://www.cdc.gov/poxvirus/monkeypox/transmission.html) | "
                "🔗 [WHO Transmission](https://www.who.int/news-room/questions-and-answers/item/monkeypox)"
            )
        else:
            # Use a predefined response when no FAQ match is found
            response_text = (
                "🔄 *How Mpox Spreads:*\n\n"
                "• Direct contact with infectious rash/scabs\n"
                "• Respiratory secretions during prolonged face-to-face contact\n"
                "• Contact with contaminated objects\n"
                "• From pregnant person to fetus\n\n"
                "✅ *Detailed Information:*\n"
                "🔗 [CDC Transmission](https://www.cdc.gov/poxvirus/monkeypox/transmission.html) | "
                "🔗 [WHO Transmission](https://www.who.int/news-room/questions-and-answers/item/monkeypox)"
            )
        
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "info", {
            "category": "transmission"
        })
        return

    # ===== PRIORITY 8: Prevention Queries =====
    if any(kw in lower_text for kw in ["prevent", "avoid", "protection", "safe"]):
        faq_answer, faq_score = faq_match(user_text, threshold=0.6)  # Lower threshold for prevention
        
        if faq_answer:
            summary = get_short_answer(faq_answer)
            response_text = (
                "🛡️ *Prevention Guide:*\n\n"
                f"_{summary}_\n\n"
                "✅ *Trusted Sources:*\n"
                "🔗 [CDC Prevention](https://www.cdc.gov/poxvirus/monkeypox/prevention.html) | "
                "🔗 [WHO Protection](https://www.who.int/news-room/questions-and-answers/item/monkeypox)"
            )
        else:
            response_text = (
                "🔍 *Key Prevention Methods:*\n\n"
                "• Avoid close contact with infected people\n"
                "• Practice good hand hygiene\n"
                "• Use PPE when caring for patients\n"
                "• Isolate if experiencing symptoms\n\n"
                "✅ *Detailed Guidelines:*\n"
                "🔗 [CDC Prevention](https://www.cdc.gov/poxvirus/monkeypox/prevention.html) | "
                "🔗 [WHO Protection](https://www.who.int/news-room/questions-and-answers/item/monkeypox)"
            )
        
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "info", {
            "category": "prevention"
        })
        return
    
    # ===== PRIORITY 9: Transmission Scenarios =====
    if is_transmission_scenario(user_text):
        # First try scenario classification
        label, explanation, reason, url, confidence = classify_scenario(user_text)
        
        if confidence > 0.65:  # Valid scenario match
            response = (
                f"🔍 *Transmission Risk Assessment:*\n\n"
                f"• **Scenario:** {user_text}\n"
                f"• **Risk Level:** {label}\n"
                f"• **Explanation:** {explanation}\n"
                f"• **Reason:** {reason}\n\n"
                f"✅ *Trusted Sources:*\n"
                f"🔗 [CDC Transmission Guide]({url})"
            )
            await update.message.reply_text(response, parse_mode="Markdown")
            return
        
        # Fallback to FAQ if scenario match is weak
        faq_answer, faq_score = faq_match(user_text, threshold=0.5)
        if faq_answer:
            response_text = (
                "🔄 *Transmission Facts:*\n\n"
                f"{get_short_answer(faq_answer)}\n\n"
                "✅ *Trusted Sources:*\n"
                "🔗 [CDC Transmission Guide](https://www.cdc.gov/poxvirus/monkeypox/transmission.html)"
            )
            await update.message.reply_text(response_text, parse_mode="Markdown")
            return
        
        # Ultimate fallback
        response_text = (
            "🔍 *General Transmission Info:*\n\n"
            "Mpox spreads through:\n"
            "• Direct contact with infectious rash\n"
            "• Respiratory secretions during prolonged contact\n"
            "• Contaminated objects (less common)\n\n"
            "✅ *Detailed Guidelines:*\n"
            "🔗 [CDC Transmission](https://www.cdc.gov/poxvirus/monkeypox/transmission.html)"
        )
        await update.message.reply_text(response_text, parse_mode="Markdown")
        return
    
    # ===== PRIORITY 10: Greetings =====
    if is_greeting(user_text):
        # Add conversational response option
        conversational_responses = [
            "😊 I'm just a bot, but I'm functioning well! How can I help with monkeypox info today?",
            "🤖 I'm a chatbot, so I don't have feelings, but I'm ready to discuss monkeypox!",
            "👋 I'm here and ready to help with any monkeypox questions you have!"
        ]
        
        if "how are you" in lower_text:
            await update.message.reply_text(random.choice(conversational_responses))
        else:
            await update.message.reply_text(random_response("greeting"))
        return  # No context storage for greetings

    # ===== PRIORITY 11: Casual Replies =====
    if is_casual_thanks(message_text):  # Use original text for thanks detection
        response_text = random_response("casual_reply")
        await update.message.reply_text(response_text)
        return  # No context storage for casual replies

    # ===== PRIORITY 12: News Requests =====
    if is_news_request(user_text):
        news_list = fetch_monkeypox_news()
        if not news_list:
            await update.message.reply_text("🚫 Couldn't fetch the latest news right now. Please try again later.")
            return

        title, news_url = news_list[0]
        response_text = (
            f"📰 *Latest Headline:*\n{title}\n"
            f"🔗 [Read more]({news_url})"
        )
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "news", {
            "title": title,
            "url": news_url
        })
        return

    # ===== PRIORITY 13: General FAQ Queries =====
    if is_general_question(user_text):
        # Double-check for vague references
        if is_vague_reference(user_text):
            return await handle_vague_query(update, context)
            
        faq_answer, faq_score = faq_match(user_text)
        if faq_answer:
            summary = get_short_answer(faq_answer)
            response_text = (
                "📘 *Informational Answer:*\n\n"
                f"_{summary}_\n\n"
                "✅ *For more details, check:* \n"
                "🔗 [CDC Mpox FAQ](https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html) | "
                "🔗 [WHO Mpox Overview](https://www.who.int/health-topics/monkeypox)"
            )
        else:
            response_text = (
                "🤔 *Hmm... I couldn't find an exact answer for that.*\n\n"
                "🔍 *But it's always best to refer to trusted health sources!* \n"
                "✅ [CDC Mpox FAQ](https://www.cdc.gov/poxvirus/monkeypox/clinicians/faq.html) | "
                "[WHO Mpox Overview](https://www.who.int/health-topics/monkeypox)"
            )
        await update.message.reply_text(response_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "faq", {
            "question": user_text,
            "answer": faq_answer,
            "summary": summary
        })
        return

    # ===== PRIORITY 14: Fallback Classification =====
    try:
        label, explanation_text, reason_text, url, _ = classify_text(user_text)
        if label.lower() == "invalid input":
            await update.message.reply_text(
                "⚠️ Sorry, I couldn't understand that. Please ask or state something clearly.",
                parse_mode="Markdown"
            )
            return
            
        result_text = (
            f"🤖 Prediction: *{label}*\n"
            f"📖 Explanation: {explanation_text}\n"
            f"📝 Reason: {reason_text}"
        )
        if url:
            result_text += f"\n🔗 [Source]({url})"
            
        await update.message.reply_text(result_text, parse_mode="Markdown", disable_web_page_preview=True)
        
        # Store context
        update_user_context(user_id, user_text, "classification", {
            "label": label,
            "explanation": explanation_text,
            "reason": reason_text,
            "source_url": url
        })
        
    except Exception as e:
        logger.exception("Error during classification")
        await update.message.reply_text(
            "😕 Oops! Something went wrong while processing your request. Please try again.",
            parse_mode="Markdown"
        )

# --- Database Logging (AFTER processing) ---
    try:
        # Determine intent category
        intent_name = "unknown"
        response_text = response_text 

        if is_joke_request(user_text):
            intent_name = "greeting"
        elif is_off_topic(user_text):
            intent_name = "off_topic"
        elif is_clear_misinfo(user_text):
            intent_name = "misinfo_check"
        elif is_news_request(user_text):
            intent_name = "news_request"
        elif is_transmission_scenario(user_text):
            intent_name = "transmission_risk"
        elif "symptom" in user_text.lower():
            intent_name = "symptom_query"
        elif any(kw in user_text.lower() for kw in ["prevent", "avoid", "protection"]):
            intent_name = "prevention_info"
        elif is_greeting(user_text):
            intent_name = "greeting"
        else:
            intent_name = "general_question"

    # Log message with intent
        message_id = log_message(user_id, message_text, intent_name)
        
        # Log response - now safe
        log_response(intent_name, response_text)
        
        # Log misinformation if detected
        if 'label' in locals() and label == "FALSE ❌":
            log_misinformation(
                content=message_text,
                source_url=url if 'url' in locals() else None
            )
            
    except Exception as e:
        logger.error(f"Database logging failed: {str(e)}")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(context, update.effective_chat.id)
    input_text = update.message.text.replace('/summarize', '').strip()

    if not input_text:
        await update.message.reply_text("✍️ Please provide some text to summarize. Example:\n`/summarize Monkeypox is...`", parse_mode="Markdown")
        return

    summary = get_short_answer(input_text)
    await update.message.reply_text(f"📄 *Summary:*\n{summary}", parse_mode="Markdown")

# Setup and run bot
def main():
    init_db()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Conversation handler for context management
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
        states={CLARIFY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_clarification)]},
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
        allow_reentry=True
    )
    
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    
    # Use polling for Hugging Face Spaces
    logger.info("Starting bot in polling mode...")
    app.run_polling()

if __name__ == "__main__":
    logger.info("Starting bot on Hugging Face Spaces...")
    main()