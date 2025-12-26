# Standard libraries for file paths and system configuration
import os 
import sys

# OpenCV for image loading and preprocessing
import cv2

# Disable PaddleOCR model source checks (useful in restricted environments)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Add project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# PaddleOCR for optical character recognition
from paddleocr import PaddleOCR

# Logger utility for tracking OCR activity
from utils.logger import get_logger

# Initialize OCR logger
logger = get_logger("ocr")
logger.info("Starting OCR extraction")

# Initialize PaddleOCR engine
# GPU is disabled to ensure compatibility on all systems
ocr = PaddleOCR(
    lang="en",
    use_gpu=False,   # Force CPU usage
    use_angle_cls=True  # Enable text angle detection
)

def extract_text_from_image(image_path):
    """
    Extracts readable text from a review screenshot using OCR.
    Returns the extracted text as a single string.
    """
    try:
        # Read image from file path
        img = cv2.imread(image_path)

        # Perform OCR on the image
        result = ocr.ocr(img, cls=True)

        lines = []

        # Iterate through detected text boxes and extract text
        for page in result:
            for box in page:
                text = box[1][0]
                lines.append(text)

        # Combine all detected lines into one text block
        raw_text = "\n".join(lines)

        # Debug output to verify OCR extraction
        print("\n========== RAW OCR TEXT ==========")
        print(raw_text)
        print("========== END RAW OCR ==========\n")

        # Return extracted text
        return raw_text

    except Exception as e:
        # Log and re-raise OCR errors
        print("OCR ERROR:", e)
        raise