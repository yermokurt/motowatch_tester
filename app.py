import gradio as gr
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageDraw, ImageFont
import tempfile
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any, Optional
import google.generativeai as genai

# For tracking
from collections import defaultdict

# Configure Gemini API
gemini_api_key = "AIzaSyCBs4TumAonKI0AodIzbl4b8Vmu9eM_r9I"  # In production, use environment variables
genai.configure(api_key=gemini_api_key)

def get_safety_analysis(stats: Dict[str, int], image_path: Optional[str] = None) -> str:
    """Generate safety analysis using Gemini AI based on detection statistics."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create a detailed prompt
        prompt = f"""You are a traffic safety analyst. Based on the following detection statistics:
        - Total Detections: {stats.get('total_detections', 0)}
        - Riders with Helmet: {stats.get('with_helmet', 0)}
        - Riders without Helmet: {stats.get('without_helmet', 0)}
        - License Plates Detected: {stats.get('license_plates', 0)}
        
        Provide a concise safety analysis and recommendations. Focus on:
        1. Helmet compliance rate
        2. Potential safety concerns
        3. Suggestions for improvement
        
        Keep the response under 100 words."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in Gemini API: {str(e)}")
        return "Safety analysis is currently unavailable. Please check your API key and internet connection."

# Download sample images and videos (optional)
sample_files = {
    'sample_1.jpg': 'https://github.com/Janno1402/Helmet-License-Plate-Detection/raw/main/Sample-Image-1.jpg',
    'sample_2.jpg': 'https://github.com/Janno1402/Helmet-License-Plate-Detection/raw/main/Sample-Image-2.jpg',
    'sample_3.jpg': 'https://github.com/Janno1402/Helmet-License-Plate-Detection/raw/main/Sample-Image-3.jpg',
    'sample_4.jpg': 'https://github.com/Janno1402/Helmet-License-Plate-Detection/raw/main/Sample-Image-4.jpg',
    'sample_5.jpg': 'https://github.com/Janno1402/Helmet-License-Plate-Detection/raw/main/Sample-Image-5.jpg',
    'traffic_violation.mp4': 'https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/raw/master/Resources/input/input.mp4'  # Traffic violation video
}

for filename, url in sample_files.items():
    if not Path(filename).exists():
        try:
            torch.hub.download_url_to_file(url, filename)
        except:
            print(f"Could not download {filename}")

# Initialize model and tracking
model = YOLO("best.pt")

# Tracking variables
track_history = defaultdict(lambda: [])
violations = defaultdict(int)


def process_image(image_path: str, conf_threshold: float = 0.4, iou_threshold: float = 0.5, 
                  image_size: int = 640, enable_tracking: bool = False) -> Tuple[Image.Image, Dict]:
    """Process a single image and return annotated image and statistics."""
    # Process image
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        verbose=False
    )
    
    # Get results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Initialize statistics with additional metrics
    total_riders = int(sum((class_ids == 0) | (class_ids == 1)))
    helmet_compliance = 0 if total_riders == 0 else int(sum(class_ids == 0) / total_riders * 100)
    
    stats = {
        'total_detections': len(boxes),
        'with_helmet': int(sum(class_ids == 0)),
        'without_helmet': int(sum(class_ids == 1)),
        'license_plates': int(sum(class_ids == 2)),
        'helmet_compliance': helmet_compliance,
        'total_riders': total_riders,
        'violation_rate': 0 if total_riders == 0 else int((sum(class_ids == 1) / total_riders) * 100)
    }
    
    # Create annotated image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{'Helmet' if class_id == 0 else 'No Helmet' if class_id == 1 else 'License Plate'} {score:.2f}"
        
        # Draw rectangle
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255) if class_id == 1 else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255))
    
    return img, stats

def process_video(video_path: str, conf_threshold: float = 0.4, iou_threshold: float = 0.5, 
                 image_size: int = 640, enable_tracking: bool = True) -> str:
    """Process a video file and return path to the output video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file."
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = model.track(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            persist=True,
            verbose=False
        )
        
        # Get tracking results
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Update tracking history and detect violations
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id == 1:  # No helmet
                    violations[track_id] += 1
                    if violations[track_id] > 10:  # If no helmet for 10 consecutive frames
                        # Draw warning
                        cv2.putText(frame, "SAFETY VIOLATION: NO HELMET!", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(results[0].plot())
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path

def process_input(input_data, input_type, conf_threshold, iou_threshold, image_size, enable_tracking):
    """Process input based on its type (image or video)."""
    if input_type == "image":
        if isinstance(input_data, str):
            img_path = input_data
        else:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img_path = temp_file.name
            input_data.save(img_path)
        
        result_img, stats = process_image(
            img_path, conf_threshold, iou_threshold, image_size, enable_tracking
        )
        
        # Generate safety analysis
        safety_analysis = get_safety_analysis(stats)
        
        # Create statistics text with safety analysis
        stats_text = f"""
        🚦 Detection Results:
        - Total Detections: {stats['total_detections']}
        - With Helmet: {stats['with_helmet']}
        - Without Helmet: {stats['without_helmet']}
        - License Plates: {stats['license_plates']}
        
        🔍 Safety Analysis:
        {safety_analysis}
        """
        
        return result_img, stats_text, None
    
    elif input_type == "video":
        if isinstance(input_data, str):
            video_path = input_data
        else:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = temp_file.name
            input_data.save(video_path)
        
        output_path = process_video(
            video_path, conf_threshold, iou_threshold, image_size, enable_tracking
        )
        
        return None, "Video processing complete!", output_path
    
    return None, "Unsupported input type", None


# Define Gradio interface components
with gr.Blocks(title="AI-Powered Helmet & License Plate Detection") as demo:
    gr.Markdown("""
    # 🛵 AI-Powered Helmet & License Plate Detection
    
    This application uses YOLOv8 to detect motorcyclists with/without helmets and license plates in images and videos.
    """)
    
    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="filepath", label="Upload Image")
                    video_input = gr.Video(visible=False)
                    
                    with gr.Row():
                        conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, 
                                             label="Confidence Threshold")
                        iou_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, 
                                            label="IOU Threshold")
                    
                    image_size = gr.Slider(minimum=320, maximum=1280, value=640, step=32, 
                                        label="Image Size")
                    
                    process_btn = gr.Button("Process", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Detection Results", type="pil")
                    stats_output = gr.Textbox(label="Detection Statistics")
                    video_output = gr.Video(visible=False)
        
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video", examples=[["traffic_violation.mp4"]])
                    image_input = gr.Image(visible=False)
                    
                    with gr.Row():
                        conf_slider_vid = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, 
                                                 label="Confidence Threshold")
                        iou_slider_vid = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, 
                                                label="IOU Threshold")
                    
                    image_size_vid = gr.Slider(minimum=320, maximum=1280, value=640, step=32, 
                                            label="Processing Frame Size")
                    
                    process_vid_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    stats_output_vid = gr.Textbox(label="Processing Status")
                    output_image = gr.Image(visible=False)
    
    # Connect the process buttons to their respective functions
    process_btn.click(
        fn=process_input,
        inputs=[
            image_input,
            gr.Number(value="image", visible=False),
            conf_slider,
            iou_slider,
            image_size,
            gr.Checkbox(value=True, visible=False)
        ],
        outputs=[output_image, stats_output, video_output]
    )
    
    process_vid_btn.click(
        fn=process_input,
        inputs=[
            video_input,
            gr.Number(value="video", visible=False),
            conf_slider_vid,
            iou_slider_vid,
            image_size_vid,
            gr.Checkbox(value=True, visible=False)
        ],
        outputs=[output_image, stats_output_vid, video_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, share=True)
