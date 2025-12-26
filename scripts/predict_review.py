# ===================== SYSTEM & PATH SETUP =====================

import sys
import os

# Add project root directory so internal modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Paths to trained ML model and vectorizer files
MODELPATH = os.path.join("models", "review_model_final.pkl")
VECTPATH = os.path.join("models", "vectorizer_final.pkl")

# Text preprocessing function
from scripts.preprocess import clean_text

# Sentiment analyzer for polarity scoring
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Model loading and numerical processing
import joblib, numpy as np

# Regular expressions and counters
import re
from collections import Counter

# Logger for prediction module
from utils.logger import get_logger

logger = get_logger("predict_review")
logger.info("Running single review prediction")

# ===================== GLOBAL MODEL OBJECTS =====================

# Cached ML model (loaded lazily)
_model = None

# Cached vectorizer
_vectorizer = None

# Sentiment analyzer instance
_analyzer = SentimentIntensityAnalyzer()

# ===================== GENERIC PHRASES (FAKE SIGNALS) =====================
# Common overused phrases that often appear in fake or promotional reviews

GENERIC_PHRASES = [
    # Basic praise
    "great product", "good product", "nice product", "excellent product",
    "amazing product", "awesome product", "perfect product", "best product",
    "very good product", "really good product", "overall good product",

    # Quality / value clichés
    "good quality", "excellent quality", "premium quality",
    "high quality product", "super quality",
    "value for money", "worth the money", "worth buying",
    "value for price", "worth the price",

    # Recommendation spam
    "highly recommended", "strongly recommended",
    "recommended to everyone", "recommended product",
    "must buy", "go for it", "blindly go for it",
    "definitely recommended",

    # Satisfaction clichés
    "very happy", "totally satisfied", "fully satisfied",
    "extremely happy", "completely satisfied",
    "happy with the purchase", "satisfied with the product",

    # Experience vagueness
    "nice experience", "great experience", "awesome experience",
    "pleasant experience", "overall experience is good",
    "good experience so far",

    # Looks / aesthetics (non-specific)
    "great display", "nice display", "beautiful display",
    "premium look", "stylish look", "looks premium",
    "nice design", "good design", "attractive design",

    # Works fine (zero detail)
    "works perfectly", "works well", "working fine",
    "does the job", "serves the purpose",
    "no issues till now", "no problems so far",

    # Overused adjectives
    "very good", "too good", "superb",
    "excellent", "awesome", "amazing",
    "fantastic", "outstanding",

    # Platform gratitude (classic fake signal)
    "thanks amazon", "thanks flipkart", "thanks meesho",
    "good service", "fast delivery", "quick delivery",
    "on time delivery", "timely delivery",

    # Star obsession
    "5 star", "five star", "giving 5 star",
    "deserves 5 star", "full five stars",

    # Repetition artifacts
    "good good", "nice nice", "very very good",
    "best best", "good good product"
]

# ===================== DETAIL SIGNALS (GENUINE SIGNALS) =====================
# Words that usually indicate real usage, experience, or reasoning

DETAIL_SIGNALS = [
    # Time & duration
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "after", "before", "since", "during", "initially", "finally",
    "first day", "first week", "after a while",

    # Usage verbs
    "used", "using", "usage", "experience", "experienced",
    "worked", "working", "tested", "testing", "tried",
    "handled", "operated", "installed", "set up",

    # Problems & failures
    "problem", "problems", "issue", "issues",
    "bug", "bugs", "error", "errors",
    "failed", "failure", "stopped", "crashed",
    "slow", "lag", "laggy", "hang", "heating",
    "overheating", "freeze", "glitch",

    # Comparison language
    "compared", "comparison", "better than", "worse than",
    "instead of", "previous", "earlier", "older",
    "compared to", "switching from",

    # Reasoning & causality
    "because", "so that", "therefore", "although",
    "however", "but", "while", "when", "if",
    "due to", "as a result",

    # Usage patterns
    "daily use", "regular use", "long term", "short term",
    "heavy use", "normal use", "light use",

    # Feature experience (not specs)
    "battery life", "battery drain", "charging time",
    "camera quality", "photo quality", "video quality",
    "sound quality", "build quality", "performance",
    "comfort", "fit", "durability", "reliability",

    # Change over time
    "improved", "reduced", "increased", "decreased",
    "fixed", "resolved", "returned", "replaced",
    "wear and tear",

    # Software / updates
    "updated", "installed", "uninstalled",
    "reset", "configured", "set up",
    "after update", "latest update",

    # Emotional contrast
    "disappointed", "satisfied", "unsatisfied",
    "happy because", "unhappy because",
    "expected more", "met expectations"
]

# ===================== SPECIFICATION INDICATORS =====================
# Phrases often copied from product descriptions

SPEC_INDICATORS = [
    "supports", "includes", "equipped with",
    "comes with", "features", "powered by",
    "built with", "offers", "provides",

    "designed to", "engineered for",
    "capable of", "optimized for",
    "rated for", "certified for",

    "available in", "configured with",
    "based on", "utilizes", "integrated with"
]

# ===================== MEASUREMENT PATTERNS =====================
# Regex patterns that detect numerical specs and measurements

MEASUREMENT_PATTERNS = [
    # Electronics
    r"\b\d+(\.\d+)?\s?(hz|khz|mah|wh|w|v|a)\b",
    r"\b\d+\s?(gb|tb|mb|kb)\b",
    r"\b\d+\s?(nm|mm|cm|m|km)\b",

    # Display / resolution
    r"\b\d+x\d+\b",
    r"\b\d+p\b",
    r"\b\d+k\b",

    # Performance
    r"\b\d+\s?(fps|seconds|secs|ms)\b",
    r"\b\d+\s?(hours|hrs|minutes|mins)\b",

    # Appliances / general
    r"\b\d+\s?(kg|g|litre|liter|ml)\b",
    r"\b\d+\s?°c\b",

    # Power / speed
    r"\b\d+\s?(rpm|bar|psi)\b"
]

# ===================== MARKETING / PROMOTIONAL WORDS =====================

MARKETING_WORDS = [
    "premium", "flagship", "top-tier", "future-proof",
    "cutting-edge", "best-in-class", "exceptional",
    "industry-leading", "powerhouse", "seamless",

    "next-generation", "revolutionary",
    "state-of-the-art", "high-end",
    "ultimate", "world-class",
    "professional-grade", "enterprise-level",

    "innovative", "advanced", "refined",
    "robust", "sleek", "elegant",
    "superior", "unmatched"
]

# Compile regex patterns for faster generic phrase matching
GENERIC_PHRASES_REGEX = [
    re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE)
    for p in GENERIC_PHRASES
]

# ===================== SHORT REVIEW HANDLING =====================
# Handles extremely short reviews separately

def handle_short_review(text: str):
    words = text.split()

    # If review is too short, classification is unreliable
    if len(words) <= 2:
        return (
            "Neutral",
            25.0,
            {
                "generic_score": round(generic_score(text) * 100, 1),
                "detail_score": 0.0,
                "sentiment": round(sentiment_score(text) * 100, 1),
                "repetition_penalty": 0.0,
                "note": "Too short to evaluate reliably"
            }
        )

    return None

# ===================== UI JUNK CLEANING =====================
# Removes non-review text like "helpful", "verified purchase", etc.

def clean_ui_junk(text: str) -> str:
    patterns = [
        r"\b\d+\s+people?\s+found\s+this\s+helpful\b",
        r"\bone\s+person\s+found\s+this\s+helpful\b",
        r"\b(one|\d+)\s+person\s+found\s+[a-z\s]{3,}\b",
        r"\bverified\s+purchase\b",
        r"\bhelpful\b",
        r"\bshare\b",
        r"\breport\b",
        r"\breviewed\s+in\s+\w+\s+on\s+\d+.*?\d{4}\b",
    ]

    # Convert text to lowercase for consistent matching
    text = text.lower()

    # Remove UI-related patterns
    for p in patterns:
        text = re.sub(p, "", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===================== GENERIC PHRASE DETECTION =====================
# Finds overused generic phrases inside a review

def find_generic_phrases(text: str):
    text_norm = re.sub(r"\s+", " ", text.lower())
    found = []

    for p in GENERIC_PHRASES:
        p_norm = re.sub(r"\s+", " ", p.lower())
        if p_norm in text_norm:
            found.append(p)

    # Remove duplicates
    return list(set(found))


# ===================== MODEL LOADING =====================
# Loads ML model and vectorizer only once (lazy loading)

def _load():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODELPATH)
        _vectorizer = joblib.load(VECTPATH)


# ===================== SENTIMENT SCORING =====================
# Uses VADER to get compound sentiment score

def sentiment_score(text: str) -> float:
    return _analyzer.polarity_scores(text)["compound"]


# ===================== GENERIC LANGUAGE SCORE =====================
# Measures how generic a review sounds

def generic_score(text: str) -> float:
    text = text.lower()
    hits = sum(1 for p in GENERIC_PHRASES if p in text)

    # Cap score at 1.0
    return min(hits / 2.0, 1.0)


# ===================== DETAIL SCORE =====================
# Measures how detailed and experience-based a review is

def detail_score(text: str) -> float:
    words = text.lower().split()
    wc = len(words)

    # HARD RULE: short reviews cannot be detailed
    if wc < 8:
        return 0.0

    # Count presence of detail-related signals
    hits = sum(1 for d in DETAIL_SIGNALS if d in text.lower())

    # Scale score by review length
    length_factor = min(wc / 40.0, 1.0)  # 40+ words = full weight

    score = (hits / 8.0) * length_factor
    return round(min(score, 1.0), 2)


# ===================== REPETITION PENALTY =====================
# Penalizes reviews that repeat the same word too often

def repetition_penalty(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())

    if len(words) < 6:
        return 0.0

    counts = Counter(words)
    most_common = counts.most_common(1)[0][1]
    repetition_ratio = most_common / len(words)

    if repetition_ratio > 0.25:
        return 0.35
    elif repetition_ratio > 0.18:
        return 0.2

    return 0.0


# ===================== PROMOTIONAL / SPEC SCORE =====================
# Detects spec-sheet narration and marketing language

def promotional_score(text: str) -> float:
    text = text.lower()
    sentences = re.split(r"[.!?]", text)

    spec_sentences = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # Detect spec-indicator phrases
        if any(ind in s for ind in SPEC_INDICATORS):
            spec_sentences += 1
            continue

        # Detect numeric measurements
        if any(re.search(p, s) for p in MEASUREMENT_PATTERNS):
            spec_sentences += 1

    spec_ratio = spec_sentences / max(len(sentences), 1)

    # Count marketing words
    marketing_hits = sum(text.count(w) for w in MARKETING_WORDS)

    score = 0.0

    # Spec-sheet narration penalty
    if spec_ratio > 0.35:
        score += 0.45
    elif spec_ratio > 0.25:
        score += 0.30

    # Marketing language density penalty
    if marketing_hits >= 3:
        score += 0.25
    elif marketing_hits >= 1:
        score += 0.15

    return min(score, 1.0)


# ===================== MAIN REVIEW ANALYSIS =====================
# Final hybrid decision combining rules, sentiment, and ML

def analyze_review(text, platform="Auto"):
    """
        FINAL HYBRID DECISION

        Returns:
          label -> "Genuine" | "Suspicious" | "Likely Fake" | "Neutral"
          confidence_percent -> float (0–100)
          meta -> diagnostic signals
    """

    # Ensure model and vectorizer are loaded
    _load()

    # -------------------- BASIC FLAGS --------------------
    original_text = text
    text_lower = text.lower() if isinstance(text, str) else ""
    has_verified_purchase = "verified purchase" in text_lower

    # -------------------- UI CLEANING --------------------
    text = clean_ui_junk(text)
    words = text.split()
    word_count = len(words)

    # Handle extremely short reviews early
    short_result = handle_short_review(text)
    if short_result:
        pred, conf, meta = short_result
        meta["word_count"] = word_count
        return pred, conf, meta

    # Special handling for short Flipkart reviews
    if platform.lower() == "flipkart" and word_count < 3:
        return (
            "Neutral",
            30.0,
            {
                "generic_score": round(generic_score(text) * 100, 1),
                "detail_score": 0.0,
                "sentiment": round(sentiment_score(text) * 100, 1),
                "repetition_penalty": 0.0,
                "note": "Short flipkart review"
            }
        )

    # 🔒 HARD GUARD 1: extremely short reviews
    if word_count < 5:
        return (
            "Neutral",
            30.0,
            {
                "generic_score": round(generic_score(text) * 100, 1),
                "detail_score": 0.0,
                "sentiment": round(sentiment_score(text) * 100, 1),
                "repetition_penalty": 0.0,
                "note": "Very short review – insufficient signal"
            }
        )

    # HARD GUARD 2: short reviews cannot be Likely Fake
    is_very_short = word_count < 7

    # Handle invalid or nearly empty input
    if not isinstance(text, str) or len(text.strip()) < 10:
        return (
            "Suspicious",
            40.0,
            {
                "generic_score": round(generic_score(text) * 100, 1),
                "detail_score": 0.0,
                "sentiment": round(sentiment_score(text) * 100, 1),
                "repetition_penalty": 0.0,
                "note": "Invalid or extremely short review"
            }
        )
    # -------------------- NLP CLEANING --------------------
    # Clean text for ML input
    cleaned = clean_text(text)

    # Fallback to raw lowercase text if cleaned version becomes too short
    if len(cleaned.split()) < 3:
        cleaned = text_lower

    # -------------------- ML PREDICTION --------------------
    # Convert text into vector form using trained vectorizer
    vec = _vectorizer.transform([cleaned])

    # If model supports probability output
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(vec)[0]
        classes = list(_model.classes_)

        # Identify index of the "genuine" class
        idx_real = classes.index(1) if 1 in classes else int(np.argmax(probs))
        ml_score = float(probs[idx_real])

    # Fallback for models without probability support
    else:
        pred = int(_model.predict(vec)[0])
        ml_score = 0.65 if pred == 1 else 0.35

    # -------------------- HEURISTIC SCORES --------------------
    # Compute individual rule-based signals
    gen = generic_score(text)

    try:
        detail = detail_score(text)
    except Exception:
        detail = 0.0

    rep_pen = repetition_penalty(text)
    sent = sentiment_score(text)
    promo = promotional_score(text)

    # Platform-specific adjustments
    if platform in ("Myntra", "Nykaa", "Meesho"):
        detail *= 0.8
        rep_pen *= 0.7

    # -------------------- PENALTIES & BONUSES --------------------
    sentiment_penalty = 0.0

    # Penalize overly positive sentiment combined with generic language
    if sent > 0.75 and gen > 0.4:
        sentiment_penalty = 0.15

    # Promotional language penalty
    promo_penalty = 0.35 * promo

    # Bonus for sufficiently detailed reviews
    detail_bonus = 0.15 if detail >= 0.5 else 0.0

    # -------------------- FINAL SCORE --------------------
    # Combine ML score with rule-based penalties and bonuses
    final_score = (
        (0.55 * ml_score)
        - (0.25 * gen)
        - rep_pen
        - sentiment_penalty
        - promo_penalty
        + detail_bonus
    )

    # Clamp score between 0 and 1
    final_score = max(0.0, min(final_score, 1.0))

    # -------------------- METADATA --------------------
    # Diagnostic values returned for explanation and UI display
    meta = {
        "generic_phrases": find_generic_phrases(text),
        "generic_score": round(gen * 100, 1),
        "detail_score": round(detail * 100, 1),
        "sentiment": round(sent * 100, 1),
        "repetition_penalty": round(rep_pen * 100, 1),
        "promotional_score": round(promo * 100, 1),
    }
    
    # SHORT BUT MEANINGFUL GENUINE REVIEW
    # Handles concise yet informative genuine reviews
    if (
        word_count >= 6
        and detail < 0.25
        and gen <= 0.3
        and sent >= 0.4
        and final_score >= 0.45
    ):
        return (
            "Genuine",
            round(min(70.0, final_score * 100), 2),
            meta
        )

    # -------------------- FINAL LABEL DECISION --------------------
    # Suspicious: high sentiment + generic language
    if detail < 0.25 and gen > 0.3 and sent > 0.85:
        prediction = "Suspicious"
        confidence = max(35.0, final_score * 100)

    # Genuine: strong final score
    elif final_score >= 0.65:
        prediction = "Genuine"
        confidence = min(90.0, final_score * 100)

    # Likely Fake or Suspicious based on confidence and length
    elif final_score <= 0.35:
        if word_count <= 15:
            prediction = "Suspicious"
            confidence = min(65.0, max(40.0, (1 - final_score) * 100))
        else:
            prediction = "Likely Fake"
            confidence = min(85.0, (1 - final_score) * 100)

    # Default case
    else:
        prediction = "Suspicious"
        confidence = min(65.0, max(35.0, final_score * 100))

    # -------------------- VERIFIED PURCHASE OVERRIDE --------------------
    # Reduce severity if review is marked as verified purchase
    if has_verified_purchase and prediction == "Likely Fake":
        prediction = "Suspicious"
        confidence *= 0.85

    # Return final decision
    return prediction, round(confidence, 2), meta