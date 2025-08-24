import easyocr
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import re
import os

class SimpleLicensePlateOCR:
    def __init__(self):
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        try:
            print("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("EasyOCR initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            return False
    
    def preprocess_image(self, image):
        try:
            if isinstance(image, Image.Image):
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img = image
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            enhanced = cv2.equalizeHist(gray)
            
            kernel = np.ones((1,1), np.uint8)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            return enhanced
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return np.array(image) if isinstance(image, Image.Image) else image
    
    def extract_text(self, image):
        if self.reader is None:
            return "OCR Reader not initialized"
        
        try:
            processed_img = self.preprocess_image(image)
            
            results = self.reader.readtext(processed_img, detail=False, paragraph=False)
            
            if not results:
                if isinstance(image, Image.Image):
                    img_array = np.array(image)
                else:
                    img_array = image
                results = self.reader.readtext(img_array, detail=False, paragraph=False)
            
            if results:
                text = ' '.join(results).strip()
                cleaned_text = self.clean_text(text)
                print(f"OCR Results: Raw='{text}' Cleaned='{cleaned_text}'")
                return cleaned_text if cleaned_text else text
            else:
                print("No text detected by OCR")
                return "No text detected"
                
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return f"OCR Error: {str(e)}"
    
    def clean_text(self, text):
        if not text:
            return ""
        
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 3:
            return text
        
        text = text.replace(' ', '')
        
        return text

ocr_instance = SimpleLicensePlateOCR()

def extract_license_plate_text(image):
    try:
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                return "Image file not found"
        
        result = ocr_instance.extract_text(image)
        print(f"Final OCR result: {result}")
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
        print("No test image found. Place a license plate image as 'test_plate.jpg' to test.")