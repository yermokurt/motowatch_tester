import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import os
from typing import Dict, List, Optional, Union
import requests
from io import BytesIO

class AdvancedLicensePlateOCR:
    def __init__(self):
        self.models = {
            "trocr_license": {
                "name": "TrOCR License Plates (Recommended)",
                "model_id": "DunnBC22/trocr-base-printed_license_plates_ocr",
                "type": "transformers",
                "processor": None,
                "model": None,
                "loaded": False,
                "description": "Specialized TrOCR model trained on license plates"
            },
            "detr_license": {
                "name": "DETR License Plate Detection + OCR",
                "model_id": "nickmuchi/detr-resnet50-license-plate-detection",
                "type": "object_detection",
                "processor": None,
                "model": None,
                "loaded": False,
                "description": "End-to-end detection and recognition"
            },
            "yolo_license": {
                "name": "YOLO License Plate (Fast)",
                "model_id": "keremberke/yolov5n-license-plate",
                "type": "yolo",
                "processor": None,
                "model": None,
                "loaded": False,
                "description": "Fast YOLO-based license plate detection"
            },
            "trocr_base": {
                "name": "TrOCR Base (General)",
                "model_id": "microsoft/trocr-base-printed",
                "type": "transformers",
                "processor": None,
                "model": None,
                "loaded": False,
                "description": "General purpose OCR model"
            },
            "easyocr": {
                "name": "EasyOCR (Fallback)",
                "model_id": "easyocr",
                "type": "easyocr",
                "processor": None,
                "model": None,
                "loaded": False,
                "description": "Traditional OCR approach"
            }
        }
        
        self.current_model = "trocr_license"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def list_available_models(self) -> Dict[str, Dict]:
        return {
            key: {
                "name": model["name"],
                "description": model["description"],
                "type": model["type"],
                "loaded": model["loaded"]
            }
            for key, model in self.models.items()
        }
    
    def load_model(self, model_key: str) -> bool:
        if model_key not in self.models:
            print(f"Model {model_key} not found")
            return False
        
        model_info = self.models[model_key]
        
        if model_info["loaded"]:
            print(f"Model {model_info['name']} already loaded")
            return True
        
        try:
            print(f"Loading {model_info['name']}...")
            
            if model_info["type"] == "transformers":
                model_info["processor"] = AutoProcessor.from_pretrained(model_info["model_id"])
                model_info["model"] = AutoModelForVision2Seq.from_pretrained(model_info["model_id"])
                model_info["model"].to(self.device)
                
            elif model_info["type"] == "object_detection":
                try:
                    model_info["model"] = pipeline(
                        "object-detection",
                        model=model_info["model_id"],
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    print(f"Failed to load as pipeline, trying alternative: {e}")
                    model_info["processor"] = AutoProcessor.from_pretrained(model_info["model_id"])
                    model_info["model"] = AutoModelForVision2Seq.from_pretrained(model_info["model_id"])
                    model_info["model"].to(self.device)
                    
            elif model_info["type"] == "yolo":
                try:
                    from ultralytics import YOLO
                    model_info["model"] = YOLO(model_info["model_id"])
                except Exception as e:
                    print(f"YOLO model loading failed: {e}")
                    return False
                    
            elif model_info["type"] == "easyocr":
                try:
                    import easyocr
                    model_info["model"] = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                except Exception as e:
                    print(f"EasyOCR loading failed: {e}")
                    return False
            
            model_info["loaded"] = True
            self.current_model = model_key
            print(f"✅ Successfully loaded {model_info['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_info['name']}: {e}")
            return False
    
    def preprocess_image_advanced(self, image: Image.Image) -> List[Image.Image]:
        variants = []
        
        try:
            original = image.copy()
            variants.append(original)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            enhancer = ImageEnhance.Contrast(image)
            high_contrast = enhancer.enhance(2.5)
            variants.append(high_contrast)
            
            sharpened = high_contrast.filter(ImageFilter.SHARPEN)
            variants.append(sharpened)
            
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            clahe_pil = Image.fromarray(clahe_img).convert('RGB')
            variants.append(clahe_pil)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_pil = Image.fromarray(binary).convert('RGB')
            variants.append(binary_pil)
            
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            denoised_pil = Image.fromarray(denoised).convert('RGB')
            variants.append(denoised_pil)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            variants = [image]
        
        return variants
    
    def extract_with_trocr(self, image: Image.Image, model_key: str) -> str:
        model_info = self.models[model_key]
        
        if not model_info["loaded"]:
            if not self.load_model(model_key):
                return "Model loading failed"
        
        try:
            processor = model_info["processor"]
            model = model_info["model"]
            
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=50)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"TrOCR extraction error: {e}")
            return f"TrOCR Error: {str(e)}"
    
    def extract_with_easyocr(self, image: Image.Image) -> str:
        model_info = self.models["easyocr"]
        
        if not model_info["loaded"]:
            if not self.load_model("easyocr"):
                return "EasyOCR loading failed"
        
        try:
            reader = model_info["model"]
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=False, paragraph=False)
            
            if results:
                return ' '.join(results).strip()
            return "No text detected"
            
        except Exception as e:
            print(f"EasyOCR extraction error: {e}")
            return f"EasyOCR Error: {str(e)}"
    
    def extract_with_detr(self, image: Image.Image) -> str:
        model_info = self.models["detr_license"]
        
        if not model_info["loaded"]:
            if not self.load_model("detr_license"):
                return "DETR model loading failed"
        
        try:
            if hasattr(model_info["model"], '__call__'):
                results = model_info["model"](image)
                if results and len(results) > 0:
                    return f"Detected {len(results)} objects"
            else:
                return self.extract_with_trocr(image, "detr_license")
                
        except Exception as e:
            print(f"DETR extraction error: {e}")
            return f"DETR Error: {str(e)}"
    
    def clean_license_text(self, text: str) -> str:
        if not text or text.startswith(("Error:", "Failed")):
            return text
        
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        common_corrections = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2'
        }
        
        for old, new in common_corrections.items():
            if sum(c.isdigit() for c in text) > sum(c.isalpha() for c in text):
                text = text.replace(old, new)
        
        return text
    
    def extract_text_with_model(self, image: Union[Image.Image, str], 
                               model_key: Optional[str] = None,
                               use_preprocessing: bool = True) -> Dict:
        
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                return {"error": f"Image file not found: {image}"}
        
        if model_key is None:
            model_key = self.current_model
        
        if model_key not in self.models:
            return {"error": f"Unknown model: {model_key}"}
        
        result = {
            "model_used": self.models[model_key]["name"],
            "model_key": model_key,
            "preprocessing": use_preprocessing,
            "extractions": [],
            "best_result": "",
            "confidence": 0.0
        }
        
        try:
            images_to_process = self.preprocess_image_advanced(image) if use_preprocessing else [image]
            
            for i, processed_img in enumerate(images_to_process):
                try:
                    if self.models[model_key]["type"] == "transformers":
                        raw_text = self.extract_with_trocr(processed_img, model_key)
                    elif self.models[model_key]["type"] == "object_detection":
                        raw_text = self.extract_with_detr(processed_img)
                    elif self.models[model_key]["type"] == "easyocr":
                        raw_text = self.extract_with_easyocr(processed_img)
                    else:
                        raw_text = "Unsupported model type"
                    
                    cleaned_text = self.clean_license_text(raw_text)
                    
                    extraction = {
                        "step": i,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "length": len(cleaned_text) if cleaned_text else 0
                    }
                    
                    result["extractions"].append(extraction)
                    
                    if cleaned_text and not cleaned_text.startswith(("Error:", "Failed")):
                        if len(cleaned_text) > len(result["best_result"]):
                            result["best_result"] = cleaned_text
                            result["confidence"] = 0.8 + (len(cleaned_text) * 0.02)
                    
                except Exception as e:
                    print(f"Error processing image variant {i}: {e}")
                    continue
            
            if not result["best_result"]:
                if result["extractions"]:
                    result["best_result"] = result["extractions"][0].get("raw_text", "No text found")
                    result["confidence"] = 0.3
                else:
                    result["best_result"] = "No text extracted"
                    result["confidence"] = 0.0
            
            return result
            
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

advanced_ocr = AdvancedLicensePlateOCR()

def get_available_models():
    return advanced_ocr.list_available_models()

def set_ocr_model(model_key: str) -> bool:
    return advanced_ocr.load_model(model_key)

def extract_license_plate_text_advanced(image: Union[Image.Image, str], 
                                       model_key: Optional[str] = None) -> str:
    try:
        result = advanced_ocr.extract_text_with_model(image, model_key)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return result.get("best_result", "No text found")
        
    except Exception as e:
        return f"Error: {str(e)}"

def get_detailed_analysis(image: Union[Image.Image, str], 
                         model_key: Optional[str] = None) -> Dict:
    return advanced_ocr.extract_text_with_model(image, model_key)

if __name__ == "__main__":
    print("Advanced License Plate OCR System")
    print("=" * 40)
    
    models = get_available_models()
    print("Available models:")
    for key, info in models.items():
        status = "✅" if info["loaded"] else "⚪"
        print(f"{status} {key}: {info['name']} - {info['description']}")
    
    print("\nRecommended models (in order):")
    print("1. trocr_license - Best for license plates")
    print("2. detr_license - End-to-end detection")
    print("3. easyocr - Reliable fallback")
    
    print("\nUsage:")
    print("from advanced_ocr import extract_license_plate_text_advanced, set_ocr_model")
    print("set_ocr_model('trocr_license')")
    print("text = extract_license_plate_text_advanced('license_plate.jpg')")