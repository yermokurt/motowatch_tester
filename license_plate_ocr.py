import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import easyocr
import os
from typing import List, Dict, Optional, Union

class LicensePlateOCR:
    def __init__(self):
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_trocr_model(self):
        try:
            print("Loading TrOCR model...")
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_model.to(self.device)
            print(f"TrOCR model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
            return False
    
    def load_easyocr_model(self):
        try:
            print("Loading EasyOCR model...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR model loaded")
            return True
        except Exception as e:
            print(f"Error loading EasyOCR model: {e}")
            return False
    
    def preprocess_license_plate(self, image: Image.Image) -> List[Image.Image]:
        processed_images = []
        
        try:
            original = image.copy()
            processed_images.append(original)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            enhancer = ImageEnhance.Contrast(image)
            high_contrast = enhancer.enhance(2.0)
            processed_images.append(high_contrast)
            
            enhancer = ImageEnhance.Sharpness(high_contrast)
            sharpened = enhancer.enhance(2.0)
            processed_images.append(sharpened)
            
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            clahe_pil = Image.fromarray(clahe_img).convert('RGB')
            processed_images.append(clahe_pil)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_pil = Image.fromarray(binary).convert('RGB')
            processed_images.append(binary_pil)
            
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            denoised_pil = Image.fromarray(denoised).convert('RGB')
            processed_images.append(denoised_pil)
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            processed_images = [image]
        
        return processed_images
    
    def extract_text_trocr(self, image: Image.Image) -> str:
        if self.trocr_processor is None or self.trocr_model is None:
            if not self.load_trocr_model():
                return ""
        
        try:
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values, max_length=50)
            
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in TrOCR extraction: {e}")
            return ""
    
    def extract_text_easyocr(self, image: Image.Image) -> str:
        if self.easyocr_reader is None:
            if not self.load_easyocr_model():
                return ""
        
        try:
            img_array = np.array(image)
            results = self.easyocr_reader.readtext(img_array, detail=0, paragraph=False)
            
            if results:
                text = ' '.join(results)
                return text.strip()
            return ""
            
        except Exception as e:
            print(f"Error in EasyOCR extraction: {e}")
            return ""
    
    def clean_license_plate_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        common_mistakes = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8',
            'G': '6', 'Z': '2', 'T': '7'
        }
        
        for mistake, correction in common_mistakes.items():
            if len([c for c in text if c.isdigit()]) > len([c for c in text if c.isalpha()]):
                text = text.replace(mistake, correction)
        
        return text
    
    def validate_license_plate_format(self, text: str) -> bool:
        if not text or len(text) < 4:
            return False
        
        common_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # XX00XX0000
            r'^[A-Z]{3}\d{4}$',               # XXX0000
            r'^[A-Z]{2}\d{4}$',               # XX0000
            r'^\d{3}[A-Z]{3}$',               # 000XXX
            r'^[A-Z]\d{3}[A-Z]{3}$',          # X000XXX
            r'^[A-Z]{2}\d{2}[A-Z]\d{3}$',     # XX00X000
        ]
        
        text_clean = text.replace(' ', '').replace('-', '')
        
        for pattern in common_patterns:
            if re.match(pattern, text_clean):
                return True
        
        if 4 <= len(text_clean) <= 10:
            alpha_count = sum(c.isalpha() for c in text_clean)
            digit_count = sum(c.isdigit() for c in text_clean)
            if alpha_count > 0 and digit_count > 0:
                return True
        
        return False
    
    def extract_license_plate_text(self, image: Union[Image.Image, str, np.ndarray], 
                                 use_preprocessing: bool = True) -> Dict[str, any]:
        
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    return {"error": f"Image file not found: {image}"}
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return {"error": f"Unsupported image type: {type(image)}"}
            
            if image.size[0] == 0 or image.size[1] == 0:
                return {"error": "Image has zero dimensions"}
            
            results = {
                "original_image_size": image.size,
                "preprocessing_used": use_preprocessing,
                "extractions": [],
                "best_result": "",
                "confidence_score": 0.0,
                "is_valid_format": False
            }
            
            images_to_process = self.preprocess_license_plate(image) if use_preprocessing else [image]
            
            all_texts = []
            
            for i, processed_img in enumerate(images_to_process):
                try:
                    trocr_text = self.extract_text_trocr(processed_img)
                    easyocr_text = self.extract_text_easyocr(processed_img)
                    
                    trocr_clean = self.clean_license_plate_text(trocr_text)
                    easyocr_clean = self.clean_license_plate_text(easyocr_text)
                    
                    extraction_result = {
                        "preprocessing_step": i,
                        "trocr_raw": trocr_text,
                        "trocr_clean": trocr_clean,
                        "easyocr_raw": easyocr_text,
                        "easyocr_clean": easyocr_clean,
                        "trocr_valid": self.validate_license_plate_format(trocr_clean),
                        "easyocr_valid": self.validate_license_plate_format(easyocr_clean)
                    }
                    
                    results["extractions"].append(extraction_result)
                    
                    if trocr_clean:
                        all_texts.append((trocr_clean, extraction_result["trocr_valid"], "trocr"))
                    if easyocr_clean:
                        all_texts.append((easyocr_clean, extraction_result["easyocr_valid"], "easyocr"))
                    
                except Exception as e:
                    print(f"Error processing image variant {i}: {e}")
                    continue
            
            if all_texts:
                valid_texts = [t for t in all_texts if t[1]]
                if valid_texts:
                    best_text = max(valid_texts, key=lambda x: len(x[0]))
                    results["best_result"] = best_text[0]
                    results["confidence_score"] = 0.9
                    results["is_valid_format"] = True
                    results["best_method"] = best_text[2]
                else:
                    longest_text = max(all_texts, key=lambda x: len(x[0]))
                    results["best_result"] = longest_text[0]
                    results["confidence_score"] = 0.6
                    results["is_valid_format"] = False
                    results["best_method"] = longest_text[2]
            else:
                results["error"] = "No text could be extracted from the image"
            
            return results
            
        except Exception as e:
            return {"error": f"Error in license plate extraction: {e}"}

def extract_license_plate_text(image_path_or_pil: Union[str, Image.Image]) -> str:
    ocr = LicensePlateOCR()
    result = ocr.extract_license_plate_text(image_path_or_pil)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return result.get("best_result", "No text found")

def get_detailed_license_plate_analysis(image_path_or_pil: Union[str, Image.Image]) -> Dict:
    ocr = LicensePlateOCR()
    return ocr.extract_license_plate_text(image_path_or_pil)

if __name__ == "__main__":
    ocr_engine = LicensePlateOCR()
    
    test_image_path = "license_plate_sample.jpg"
    
    if os.path.exists(test_image_path):
        print("Testing license plate OCR...")
        
        result = ocr_engine.extract_license_plate_text(test_image_path)
        
        print(f"Best Result: {result.get('best_result', 'No text found')}")
        print(f"Valid Format: {result.get('is_valid_format', False)}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        
        print("\nDetailed Results:")
        for i, extraction in enumerate(result.get('extractions', [])):
            print(f"  Step {i}:")
            print(f"    TrOCR: {extraction['trocr_clean']} (Valid: {extraction['trocr_valid']})")
            print(f"    EasyOCR: {extraction['easyocr_clean']} (Valid: {extraction['easyocr_valid']})")
    else:
        print(f"Test image {test_image_path} not found.")
        print("Usage example:")
        print("  from license_plate_ocr import extract_license_plate_text")
        print("  text = extract_license_plate_text('your_license_plate.jpg')")
        print("  print(text)")