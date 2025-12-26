# Standard libraries for system path handling
import sys
import os

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import the single-review prediction function
from scripts.predict_review import analyze_review

# Logger utility for tracking bulk prediction activity
from utils.logger import get_logger

# Initialize logger for bulk prediction
logger = get_logger("predict_bulk")
logger.info("Bulk prediction started")


def analyze_bulk(reviews):
    """
    Analyzes multiple reviews in one call.

    Parameters:
    reviews: list[str]
        A list of review texts to be analyzed.

    Returns:
    fake_count: int
        Number of reviews predicted as Fake.
    real_count: int
        Number of reviews predicted as Genuine.
    result_list: list[dict]
        Each dictionary contains:
        {
            'text': original review text,
            'prediction': model prediction,
            'confidence': confidence score
        }
    """
    fake = 0      # Counter for fake reviews
    real = 0      # Counter for genuine reviews
    results = []  # Store detailed results for each review

    # Process each review one by one
    for r in reviews:
        # Run ML model on the review text
        p, c = analyze_review(r)

        # Store prediction result
        results.append({
            'text': r,
            'prediction': p,
            'confidence': c
        })

        # Update counters based on prediction
        if p == "Fake":
            fake += 1
        elif p == "Genuine":
            real += 1

    # Return summary counts and detailed results
    return fake, real, results