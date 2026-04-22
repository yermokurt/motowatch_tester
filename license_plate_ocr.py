import easyocr
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import re
import os

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

class SimpleLicensePlateOCR:
    def __init__(self):
        self.reader = None
        self.paddle_reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        # 1. Try PaddleOCR (User Preferred)
        if PADDLE_AVAILABLE:
            try:
                print("Initializing PaddleOCR...")
                # use_gpu=True for GTX 1650 Ti
                self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
                print("PaddleOCR initialized successfully")
                return True
            except Exception as e:
                print(f"Failed to initialize PaddleOCR: {e}")

        # 2. Fallback to EasyOCR
        try:
            print("Initializing EasyOCR (Fallback)...")
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print("EasyOCR initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            return False
    
    def get_variants(self, image):
        """Generate multiple image variants to improve OCR chances."""
        variants = []
        try:
            if isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variants.append(gray)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            variants.append(clahe.apply(gray))
            
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            variants.append(sharpened)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(binary)
            
            variants.append(cv2.bitwise_not(binary))
            
            return variants
        except Exception as e:
            print(f"Variant generation error: {e}")
            return [gray] if 'gray' in locals() else []

    def score_candidate(self, text):
        """Score how well a text matches Philippine plate formats."""
        if not text: return 0
        
        patterns = [
            (r'^[A-Z]{3}\d{4}$', 100), # ABC 1234
            (r'^[A-Z]{3}\d{3}$', 90),  # ABC 123
            (r'^\d{3}[A-Z]{3}$', 95),  # 123 ABC
            (r'^[A-Z]{2}\d{5}$', 85),  # AB 12345
            (r'^[A-Z]\d{4,6}$', 70),   # Local/Utility
        ]
        
        cleaned = text.replace(' ', '').upper()
        for pattern, score in patterns:
            if re.match(pattern, cleaned):
                return score
        
        score = 10
        if any(c.isalpha() for c in cleaned) and any(c.isdigit() for c in cleaned):
            score += 20
        if 4 <= len(cleaned) <= 8:
            score += 10
            
        return score

    def refine_character(self, text):
        """Fix common OCR swaps based on likely PH plate character positions."""
        if not text or len(text) < 4: return text
        
        num_to_let = {'0':'O', '1':'I', '2':'Z', '5':'S', '8':'B', '6':'G', '7':'Z'}
        let_to_num = {'O':'0', 'I':'1', 'Z':'2', 'S':'5', 'B':'8', 'G':'6', 'Q':'0'}
        
        chars = list(text.upper().replace(' ', ''))
        
        if len(chars) >= 6:
            for i in range(3):
                if chars[i] in num_to_let: chars[i] = num_to_let[chars[i]]
            for i in range(3, len(chars)):
                if chars[i] in let_to_num: chars[i] = let_to_num[chars[i]]
        
        return "".join(chars)

    def clean_text(self, text):
        if not text:
            return ""
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)
        if len(text) < 3:
            return text
        return text

    def extract_text(self, image):
        if self.reader is None:
            return "OCR Reader not initialized"
        
        try:
            variants = self.get_variants(image)
            candidates = []
            
            for i, var in enumerate(variants):
                if self.paddle_reader:
                    # PaddleOCR variant
                    result = self.paddle_reader.ocr(var, cls=True)
                    if result and result[0]:
                        # Paddle returns [[[box], [text, score]], ...]
                        raw = "".join([line[1][0] for line in result[0]])
                        cleaned = self.clean_text(raw)
                        refined = self.refine_character(cleaned)
                        score = self.score_candidate(refined)
                        candidates.append({"text": refined, "score": score, "variant": i})
                elif self.reader:
                    # EasyOCR fallback
                    results = self.reader.readtext(var, detail=False, paragraph=False)
                    if results:
                        raw = "".join(results)
                        cleaned = self.clean_text(raw)
                        refined = self.refine_character(cleaned)
                        score = self.score_candidate(refined)
                        candidates.append({"text": refined, "score": score, "variant": i})
            
            if not candidates:
                return "No text detected"
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best = candidates[0]
            
            print(f"OCR Best Candidate: {best['text']} (Score: {best['score']}, Variant: {best['variant']})")
            return best["text"]
                
        except Exception as e:
            print(f"OCR ensemble error: {e}")
            return f"OCR Error: {str(e)}"

ocr_instance = SimpleLicensePlateOCR()

def extract_license_plate_text(image):
    try:
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                return "Image file not found"
        
        result = ocr_instance.extract_text(image)
        return result
        
    except Exception as e:
        error_msg = f"Error in text extraction: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    print("Testing License Plate OCR...")
    test_image = "test_plate.jpg"
    if os.path.exists(test_image):
        result = extract_license_plate_text(test_image)
        print(f"Test result: {result}")
    else:
        print("No test image found.")