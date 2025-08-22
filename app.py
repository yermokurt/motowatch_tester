import gradio as gr
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import requests
import json

# Download sample images
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-1.jpg?raw=true', 'sample_1.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-2.jpg?raw=true', 'sample_2.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-3.jpg?raw=true', 'sample_3.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-4.jpg?raw=true', 'sample_4.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-5.jpg?raw=true', 'sample_5.jpg')

model = YOLO("best.pt")

GEMINI_API_KEY = "AIzaSyCBs4TumAonKI0AodIzbl4b8Vmu9eM_r9I"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_gemini_response(prompt: str) -> str:
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return "Unable to get analysis at the moment."

def get_safety_analysis(stats: dict) -> str:
    prompt = f"""
    You are a traffic safety analyst. Analyze the following statistics and provide a brief safety report:
    
    Detection Results:
    - Total Detections: {stats.get('total_detections', 0)}
    - Riders with Helmet: {stats.get('with_helmet', 0)}
    - Riders without Helmet: {stats.get('without_helmet', 0)}
    - Helmet Compliance Rate: {stats.get('helmet_compliance', 0)}%
    - License Plates Detected: {stats.get('license_plates', 0)}
    
    Provide a 3-4 sentence safety analysis focusing on helmet compliance and potential safety concerns.
    """
    return get_gemini_response(prompt)

def yoloV8_func(image=None, image_size=640, conf_threshold=0.4, iou_threshold=0.5):
    print(f"Received image_size: {image_size}")
    
    if image_size is None:
        image_size = 640
    
    if not isinstance(image_size, int):
        image_size = int(image_size)
    
    imgsz = [image_size, image_size]

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False
    )
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    total_riders = int(sum((class_ids == 0) | (class_ids == 1)))
    helmet_compliance = 0 if total_riders == 0 else int(sum(class_ids == 0) / total_riders * 100)
    
    stats = {
        'total_detections': len(boxes),
        'with_helmet': int(sum(class_ids == 0)),
        'without_helmet': int(sum(class_ids == 1)),
        'license_plates': int(sum(class_ids == 2)),
        'helmet_compliance': helmet_compliance,
        'total_riders': total_riders
    }
    
    safety_analysis = get_safety_analysis(stats)
    print("\nSafety Analysis:", safety_analysis)
    
    annotated_image = results[0].plot()
    
    return annotated_image

with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Output Image")
    
    with gr.Row():
        image_size = gr.Slider(
            minimum=320, 
            maximum=1280, 
            value=640, 
            step=32, 
            label="Image Size",
            interactive=True
        )
        conf_threshold = gr.Slider(
            minimum=0.1, 
            maximum=1.0, 
            value=0.4, 
            step=0.05, 
            label="Confidence Threshold",
            interactive=True
        )
        iou_threshold = gr.Slider(
            minimum=0.1, 
            maximum=1.0, 
            value=0.5, 
            step=0.05, 
            label="IOU Threshold",
            interactive=True
        )
    
    process_btn = gr.Button("Process Image")
    process_btn.click(
        fn=yoloV8_func,
        inputs=[image_input, image_size, conf_threshold, iou_threshold],
        outputs=output_image
    )

outputs = gr.Image(type="pil", label="Output Image")

title = "YOLOv11 Motorcyclist Helmet Detection"
description = """
    This application uses YOLOv11 to detect Motorcyclists with and without Helmets in images. 
    Upload an image, adjust the confidence and IOU thresholds, and view the detection results. 
    You can customize the model's performance to fit your needs.
"""
article = """
    <h2>How It Works:</h2>
    <p>This model detects Motorcyclists with and without Helmets in images and highlights them with bounding boxes. 
    Adjust the confidence threshold to control detection accuracy and the IOU threshold for overlap sensitivity.</p>
    <p>Upload your images and try it out!</p>
"""

if __name__ == "__main__":
    demo.launch(debug=True)
