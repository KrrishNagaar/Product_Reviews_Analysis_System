# ===================== SYSTEM & PATH SETUP =====================

import sys
import os

# Add project root directory so internal modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# -------------------- NLP & TEXT PROCESSING --------------------
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Logger for preprocessing module
from utils.logger import get_logger

logger = get_logger("preprocess")
logger.debug("Cleaning text")


# ===================== NLTK SETUP =====================
# Ensure required NLTK resources are available

try:
    nltk.data.find("corpora/stopwords")    
    nltk.data.find("corpora/wordnet")
    nltk.data.find("tokenizers/punkt")
except LookupError:
    # Download missing resources if not found
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()


# ===================== OCR LINE NORMALIZATION =====================
# Cleans raw OCR output line-by-line

def normalize_ocr_lines(text: str):
    lines = []

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue

        # ❌ DO NOT touch star lines here (handled separately)
        lines.append(raw)

    # Debug output for OCR inspection
    print("\n========== NORMALIZED OCR LINES ==========")
    for i, l in enumerate(lines):
        print(f"{i:02d}: {repr(l)}")
    print("========== END NORMALIZED ==========\n")

    return lines


# ===================== STAR SYMBOL CLEANING =====================
# Removes star symbols but preserves the actual review text

def strip_star_line_content(line: str) -> str:
    """
    Removes star symbols but keeps any text after them.
    Example:
    ★★★★★ Great  -> Great
    **** battery issue -> battery issue
    """
    return re.sub(r"^[★*]{3,}\s*", "", line).strip()


# ===================== UI / PAGE NOISE CLEANING =====================
# Removes buttons, badges, emojis, and platform UI artifacts

def clean_ui_junk(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text_clean = text

    patterns = [
        # UI buttons
        r"\bhelpful\b", r"\bshare\b", r"\breport\b",
        r"\bcomment\b", r"\bfollow\b",

        # Purchase badges
        r"\bverified\s+buyer\b",
        r"\bcertified\s+buyer\b",

        # Helpful counts
        r"\b\d+\s+people?\s+found\s+this\s+helpful\b",
        r"\b\d+\s+votes?\b",

        # Dates / locations (remove entire metadata line)
        r"\breviewed\s+in\s+\w+\s+on\s+\d+.*?\d{4}\b",

        # Ratings
        r"\b\d(\.\d)?\s*star\b",
        r"\b\d/5\b",

        # Usernames (light filtering)
        r"\buser\s*\d+\b",
        r"\bkk\s+kk", r"\bkk\s+kw", r"\bkw\s+kw",

        # Emojis
        r"[❤️😍🔥👍⭐]+",
    ]

    # Apply all noise-removal patterns
    for p in patterns:
        text_clean = re.sub(p, " ", text_clean, flags=re.IGNORECASE)

    # Normalize whitespace
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    return text_clean


# ===================== AMAZON REVIEW SPLITTING =====================
# Extracts individual reviews from Amazon OCR text

def split_amazon(ocr_text):
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]

    reviews = []
    buffer = []
    collecting = False

    def is_username(line):
        # Detect typical username patterns
        return bool(re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$", line))

    def is_start(line):
        # Marks beginning of review content
        return "Reviewed in" in line or "Verified Purchase" in line

    def is_end(line):
        # Marks end of review content
        l = line.lower()
        return (
            l == "report"
            or "found this helpful" in l
        )

    for line in lines:
        # Username detected → reset buffer
        if is_username(line):
            buffer = []
            collecting = False
            continue

        # Start collecting review text
        if is_start(line):
            collecting = True
            continue

        # End of current review
        if is_end(line):
            if buffer:
                reviews.append(" ".join(buffer))
            buffer = []
            collecting = False
            continue

        # Collect review lines
        if collecting:
            buffer.append(line)

        # Capture title text before review starts
        elif not collecting and not is_username(line):
            buffer.append(line)

    # Append last buffered review if any
    if buffer:
        reviews.append(" ".join(buffer))

    # OCR-safe minimum length filter
    return [r for r in reviews if len(r.split()) >= 3]


# ===================== FLIPKART REVIEW SPLITTING =====================
# Platform-specific logic for Flipkart OCR reviews

def split_flipkart(text: str):
    reviews = []
    buffer = []
    collecting = False

    # Normalize OCR lines before processing
    lines = normalize_ocr_lines(text)

    def flush():
        review = " ".join(buffer).strip()

        # 🔒 HARD FILTER — Flipkart noise control
        if len(review.split()) < 5:
            return

        # Reject pure names
        if re.fullmatch(r"[A-Za-z ]{3,}", review):
            return

        reviews.append(review)

    def is_username(line):
        return bool(re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}$", line))

    def is_start(line: str) -> bool:
        # Detect star-based review start
        return (
            re.match(r"^[★*]{2,}\s*\d", line) or   # ***** 5.0
            re.match(r"^\d\s*\*", line)            # 5*
        )

    def is_end(line: str) -> bool:
        low = line.lower()
        return (
            "verified purchase" in low or
            re.match(r"^[★*]{2,}\s*\d", line) or
            re.match(r"^\d\s*\*", line)
        )

    def is_noise(line: str) -> bool:
        low = line.lower().strip()
        return (
            low.isdigit() or
            low in {"...", "iii", "o", ":"} or
            "page" in low or
            "next" in low or
            "back to top" in low or
            "helpful" in low
        )

    def is_strong_title(line: str) -> bool:
        # Recognize common one-line Flipkart titles
        low = line.lower().strip()
        return low in {
            "excellent", "super", "brilliant", "awesome",
            "terrific", "must buy", "just wow!",
            "perfect product!", "worth every penny",
            "classy product", "fabulous", "wonderful"
        }

    def is_section_header(line: str) -> bool:
        # Ignore section headers
        low = line.lower().strip()
        return low in {
            "overall",
            "quality",
            "design & finish",
            "design and finish",
            "ratings",
            "reviews"
        }
    # Iterate through OCR lines with index
    for idx, line in enumerate(lines):
        low = line.lower().strip()
        
        # Skip section headers like "Ratings", "Reviews", etc.
        if is_section_header(line):
            continue

        # Skip username-only lines
        if is_username(line):
            continue

        # -------- START (also handles FIRST review) --------
        # Detect start of a new review or a strong title during collection
        if is_start(line) or (collecting and is_strong_title(line)):
            if collecting and buffer:
                flush()
                buffer = []

            collecting = True
            buffer.append(line)
            continue

        # -------- END --------
        # End of a review block
        if is_end(line):
            flush()
            buffer = []
            collecting = False
            continue

        # -------- IGNORE NOISE --------
        # Skip OCR noise, page markers, or irrelevant text
        if is_noise(line):
            continue

        # Ignore patterns like "Name, City"
        if re.match(r"^[A-Za-z .]{3,},\s*[A-Za-z]{3,}$", line):
            continue

        # -------- COLLECT --------
        # Collect review content while inside a review
        if collecting:
            buffer.append(line)

    # 🔥 VERY IMPORTANT: flush last review (was missing earlier)
    if buffer:
        flush()

    return reviews


# ===================== MEESHO-SPECIFIC JUNK FILTER =====================
# Identifies meaningless or broken Meesho review fragments

def is_meesho_junk(review: str) -> bool:
    low = review.lower().strip()

    # ReadMore-only or broken tails
    if low in {"read more", "readmore", "...read more"}:
        return True

    # Only punctuation or OCR noise
    if re.fullmatch(r"[.:x]+", low):
        return True

    # Too generic continuation words
    if low.startswith(("and ", "but ", "or ", "so ")):
        return True

    return False


# Minimum words required to accept a review
MIN_WORDS = 2


# PC page-level junk patterns
PC_JUNK = (
    "people also viewed",
    "shop on app",
    "home/",
)


# ===================== SHORT REVIEW HANDLER =====================
# Handles extremely short Meesho reviews

def handle_short_review(review_text: str):
    words = review_text.split()

    if len(words) <= 2:
        return {
            "prediction": "Neutral",
            "confidence": 25.0,
            "reason": "Too short to evaluate reliably"
        }

    return None


# ===================== SENTENCE HEURISTIC =====================
# Checks if a line looks like actual review text

def looks_like_sentence(line: str) -> bool:
    words = line.split()
    return (
        len(words) >= 2
        and not re.fullmatch(r"[0-9.*]+", line)
        and not any(j in line.lower() for j in PC_JUNK)
    )


# ===================== MEESHO REVIEW SPLITTING =====================
# Extracts individual reviews from Meesho OCR text

def split_meesho(text: str):
    blocks = []
    current = []
    state = "IDLE"

    def flush(allow_empty=False):
        nonlocal current, state

        # Combine collected lines into one review
        review = " ".join(current).strip()

        # Remove trailing "read more" artifacts
        review = re.sub(
            r"\b(read\s*more|\.{2,})\b",
            "",
            review,
            flags=re.I
        ).strip()

        # Handle empty or image-only reviews
        if not review:
            if allow_empty:
                blocks.append("[IMAGE_ONLY_REVIEW]")
            current = []
            state = "IDLE"
            return

        # Reject numeric-only junk
        if re.fullmatch(r"[0-9\s:.*]+", review):
            current = []
            state = "IDLE"
            return

        # Enforce minimum word count
        if len(review.split()) >= MIN_WORDS:
            blocks.append(review)

        current = []
        state = "IDLE"

    def is_username(line: str) -> bool:
        low = line.lower().strip()
        return (
            low == "meesho user"
            or bool(re.fullmatch(r"[A-Z][a-z]+(?: [A-Za-z]+){0,2}", line))
        )

    for line in normalize_ocr_lines(text):
        raw = line.strip()
        low = raw.lower()

        # -------- HARD JUNK --------
        # Remove meaningless OCR artifacts
        if raw == "x" or raw.isupper() or "&" in raw:
            continue

        if any(j in low for j in PC_JUNK):
            if state == "BODY":
                flush()
            continue

        # -------- HELPFUL = HARD END --------
        if "helpful" in low:
            if state == "BODY":
                flush(allow_empty=True)
            continue

        # -------- USERNAME --------
        if is_username(raw):
            if state == "BODY":
                flush()
            state = "META"
            continue

        # -------- METADATA --------
        # Detect time or posting metadata
        if "posted" in low or "days ago" in low:
            state = "BODY"
            continue

        # -------- RATING (mobile + pc) --------
        if re.fullmatch(r"\d(\.\d)?\*?", raw) and state != "BODY":
            continue

        # -------- BODY (content-driven) --------
        if state in {"META", "IDLE"}:
            if looks_like_sentence(raw) or len(raw.split()) <= 2:
                state = "BODY"

        if state == "BODY":
            current.append(raw)

    # Flush remaining review if still collecting
    if state == "BODY":
        flush()

    return blocks


# ===================== REGEX PATTERNS =====================

STAR_RE = re.compile(
    r"^\s*(\d+\s*\*.*|\d+\s*$|\*+\s*.*|★+\s*.*)$",
    re.I
)

TIME_RE = re.compile(
    r"^(a\s+)?\d+\s+(day|days|month|months|year|years)\s+ago\.?$",
    re.I
)

AUTHOR_RE = re.compile(
    r"^[A-Za-z .]+(\||\s)\s*\d{1,2}\s+\w+\s+\d{4}$",
    re.I
)

NUMERIC_NOISE_RE = re.compile(
    r"^\s*(0|\d+|\d+\s*\[|ip|op|0 p|1 p|q 1)\s*$"
)

DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")

NAME_RE = re.compile(
    r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}$"
)


# ===================== MYNTRA HEADER DETECTION =====================
# Identifies header lines in Myntra review pages

def is_myntra_header(line: str) -> bool:
    l = line.lower().strip()
    return (
        "ratings & reviews" in l or
        "customer reviews" in l or
        l.startswith("<ratings") or
        l.startswith("ratings") or
        l.startswith("most helpful")
    )
# ===================== LINE CLASSIFICATION =====================
# Categorizes each OCR line into a structural type

def classify_line(line: str):
    l = line.lower().strip()

    # Star rating line (e.g., ★★★★, 5*)
    if STAR_RE.fullmatch(line):
        return "STAR"

    # Date-only line
    if DATE_RE.fullmatch(line):
        return "TIME"

    # Name-only line
    if NAME_RE.fullmatch(line):
        return "NAME"

    # Author + date combined line
    if AUTHOR_RE.match(line):
        return "AUTHOR"

    # Relative time (e.g., "2 days ago")
    if TIME_RE.match(l):
        return "TIME"

    # Numeric or meaningless OCR noise
    if NUMERIC_NOISE_RE.fullmatch(l):
        return "NOISE"

    # Very short text (likely not a full review)
    if len(l.split()) <= 3:
        return "SHORT_TEXT"

    # Default: meaningful review text
    return "TEXT"


# ===================== PC MYNTRA DETECTION =====================
# Checks whether OCR output resembles PC layout

def is_pc_myntra(lines):
    return any(re.fullmatch(r"\d", l.strip()) for l in lines)


# ===================== MYNTRA REVIEW SPLITTER =====================
# Extracts reviews from Myntra OCR text

def split_myntra(ocr_text: str):
    lines = normalize_ocr_lines(ocr_text)
    pc_mode = is_pc_myntra(lines)

    reviews = []
    buffer = []
    collecting = False

    def flush():
        nonlocal buffer, collecting

        # Combine buffered lines into one review
        review = " ".join(buffer).strip()

        # Enforce minimum word count
        if len(review.split()) >= 2:
            reviews.append(review)

        buffer = []
        collecting = False

    for raw in lines:
        low = raw.lower().strip()
        t = classify_line(raw)

        # ---------- HARD IGNORE ----------
        if not low:
            continue

        # Ignore page headers
        if is_myntra_header(raw):
            continue

        # Ignore UI noise
        if "size bought" in low or "helpful" in low:
            continue

        # Ignore buyer badge
        if "verified buyer" in low:
            continue
        
        if t == "NOISE":
            continue

        # ---------- STAR = HARD START ----------
        # Star rating marks beginning of a review
        if t == "STAR":
            if buffer:
                flush()
            collecting = True
            continue

        # ---------- AUTHOR = HARD END ----------
        # Author line ends the current review
        if t == "AUTHOR":
            if collecting:
                flush()
            continue

        # ---------- TIME ----------
        # Time behaves differently for PC vs mobile
        if t == "TIME":
            if pc_mode:
                continue  # PC layout ignores time lines
            if collecting:
                flush()
            collecting = True
            continue

        # ---------- HARD REVIEW BOUNDARIES ----------
        # Names or dates reset collection
        if t in {"NAME", "DATE"}:
            if buffer:
                flush()
            continue

        # ---------- TEXT ----------
        # Collect review content
        if collecting:
            buffer.append(raw)

    # Flush remaining review if any
    if buffer:
        flush()

    return reviews


# ===================== REVIEW VALIDATION =====================
# Minimal validation before model analysis

def is_valid_review(text):
    words = text.split()
    print("[VALIDATOR] is_valid_review | words:", len(words))
    return len(words) >= 1


# ===================== REPETITION COLLAPSING =====================
# Removes consecutive duplicate words

def _collapse_repetition(text: str) -> str:
    """
    Converts:
    'good good good product' → 'good product'
    """
    words = text.split()
    collapsed = []

    for w in words:
        if not collapsed or collapsed[-1] != w:
            collapsed.append(w)

    return " ".join(collapsed)


# ===================== FINAL NLP CLEANING =====================
# Normalizes text before ML processing

def clean_text(text: str) -> str:
    """
    NLP normalization for fake-review detection
    """
    if not isinstance(text, str):
        return ""

    # Lowercase text
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Normalize punctuation spacing
    text = re.sub(r"[! ? ]{2,}", "!", text)
    text = re.sub(r"\. {2,}", ".", text)

    # Remove punctuation symbols
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text
    words = nltk.word_tokenize(text)

    # Lemmatize and filter very short tokens
    filtered = [
        lemmatizer.lemmatize(w)
        for w in words
        if len(w) > 2
    ]

    # Collapse repeated words
    cleaned_text = _collapse_repetition(" ".join(filtered))

    # ✅ Allow 2+ word OCR reviews
    return cleaned_text if len(cleaned_text.split()) >= 2 else text.lower()


# ===================== PLATFORM DISPATCHER =====================
# Routes OCR text to platform-specific splitters

def split_reviews(text: str, platform: str):
    platform = platform.lower()

    if platform == "amazon":
        result = split_amazon(text)

        # Debug output for Amazon splitting
        print("\n[DEBUG] ===== SPLIT_AMAZON OUTPUT =====")
        for i, r in enumerate(result):
            print(f"[DEBUG] SPLIT {i}: {repr(r)} | words={len(r.split())}")
        print("[DEBUG] =================================\n")

        return result

    elif platform == "flipkart":
        return split_flipkart(text)

    elif platform == "meesho":
        return split_meesho(text)

    elif platform.lower() == "myntra":
        return split_myntra(text)

    elif platform.lower() == "nykaa":
        return split_myntra(text)

    else:
        # Safest fallback behavior
        return split_amazon(text)