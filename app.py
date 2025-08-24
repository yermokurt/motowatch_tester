import gradio as gr
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os
import cv2
import time
import zipfile
import io
from datetime import datetime

# Download sample images (optional)
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-1.jpg?raw=true', 'sample_1.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-2.jpg?raw=true', 'sample_2.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-3.jpg?raw=true', 'sample_3.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-4.jpg?raw=true', 'sample_4.jpg')
torch.hub.download_url_to_file('https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-5.jpg?raw=true', 'sample_5.jpg')

# Load model (cached for performance)
model = YOLO("best.pt")
class_names = {0: 'With Helmet', 1: 'Without Helmet', 2: 'License Plate'}

def crop_license_plates(image, detections):
    cropped_plates = []
    
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                print(f"Error: Image file not found: {image}")
                return cropped_plates
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            print(f"Error: Unsupported image type: {type(image)}")
            return cropped_plates
        
        if image.size[0] == 0 or image.size[1] == 0:
            print("Error: Image has zero dimensions")
            return cropped_plates
            
    except Exception as e:
        print(f"Error loading image: {e}")
        return cropped_plates
    
    for i, detection in enumerate(detections):
        try:
            if detection['Object'] != 'License Plate':
                continue
                
            pos_str = detection['Position'].strip('()')
            if ',' not in pos_str:
                print(f"Error: Invalid position format for detection {i}: {detection['Position']}")
                continue
                
            x1, y1 = map(int, pos_str.split(', '))
            
            dims_str = detection['Dimensions']
            if 'x' not in dims_str:
                print(f"Error: Invalid dimensions format for detection {i}: {detection['Dimensions']}")
                continue
                
            width, height = map(int, dims_str.split('x'))
            
            if width <= 0 or height <= 0:
                print(f"Error: Invalid dimensions for detection {i}: {width}x{height}")
                continue
            
            x2, y2 = x1 + width, y1 + height
            
            if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                print(f"Warning: Bounding box extends beyond image boundaries for detection {i}")
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                print(f"Error: Invalid crop coordinates for detection {i}: ({x1},{y1}) to ({x2},{y2})")
                continue
            
            cropped_plate = image.crop((x1, y1, x2, y2))
            
            if cropped_plate.size[0] == 0 or cropped_plate.size[1] == 0:
                print(f"Error: Cropped image has zero dimensions for detection {i}")
                continue
            
            cropped_plates.append({
                'image': cropped_plate,
                'confidence': detection['Confidence'],
                'position': detection['Position'],
                'crop_coords': f"({x1},{y1}) to ({x2},{y2})"
            })
            
        except ValueError as e:
            print(f"Error parsing coordinates for detection {i}: {e}")
            continue
        except Exception as e:
            print(f"Error cropping license plate {i}: {e}")
            continue
    
    return cropped_plates

def create_download_files(annotated_image, cropped_plates, detections):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("temp", exist_ok=True)
        
        annotated_path = f"temp/annotated_image_{timestamp}.jpg"
        try:
            annotated_image.save(annotated_path, quality=95)
        except Exception as e:
            print(f"Error saving annotated image: {e}")
            return None, None, []
        
        plate_paths = []
        for i, plate_data in enumerate(cropped_plates):
            try:
                plate_path = f"temp/license_plate_{i+1}_{timestamp}.jpg"
                plate_data['image'].save(plate_path, quality=95)
                plate_paths.append(plate_path)
            except Exception as e:
                print(f"Error saving license plate {i+1}: {e}")
                continue
        
        report_path = f"temp/detection_report_{timestamp}.csv"
        if detections:
            try:
                df = pd.DataFrame(detections)
                df.to_csv(report_path, index=False)
            except Exception as e:
                print(f"Error creating detection report: {e}")
                report_path = None
        
        zip_path = f"temp/detection_results_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(annotated_path):
                    zipf.write(annotated_path, f"annotated_image_{timestamp}.jpg")
                for plate_path in plate_paths:
                    if os.path.exists(plate_path):
                        zipf.write(plate_path, os.path.basename(plate_path))
                if report_path and os.path.exists(report_path):
                    zipf.write(report_path, f"detection_report_{timestamp}.csv")
        except Exception as e:
            print(f"Error creating ZIP file: {e}")
            return None, annotated_path, plate_paths
        
        return zip_path, annotated_path, plate_paths
        
    except Exception as e:
        print(f"Error in create_download_files: {e}")
        return None, None, []

def yoloV8_func(
    image=None, 
    image_size=640, 
    conf_threshold=0.4, 
    iou_threshold=0.5,
    show_stats=True,
    show_confidence=True,
    crop_plates=True
):
    # Handle NoneType for image_size
    if image_size is None:
        image_size = 640
    
    # Ensure image_size is an integer
    if not isinstance(image_size, int):
        image_size = int(image_size)
    
    # Construct imgsz as a list of two integers [width, height]
    imgsz = [image_size, image_size]

    # Make predictions
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz)

    # Get the output image with bounding boxes
    annotated_image = results[0].plot()  # This returns a PIL image
    
    # Convert to PIL if it's a numpy array
    if isinstance(annotated_image, np.ndarray):
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    
    # Extract detection information
    boxes = results[0].boxes
    detections = []
    
    if boxes is not None and len(boxes) > 0:
        for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls)
            confidence = float(conf)
            label = class_names.get(class_id, f"Class {class_id}")
            
            detections.append({
                "Object": label,
                "Confidence": f"{confidence:.2f}",
                "Position": f"({int(x1)}, {int(y1)})",
                "Dimensions": f"{int(x2-x1)}x{int(y2-y1)}"
            })
    
    # Crop license plates if requested
    cropped_plates = []
    license_plate_gallery = []
    download_files = None
    
    if crop_plates and detections:
        try:
            cropped_plates = crop_license_plates(image, detections)
            license_plate_gallery = [plate_data['image'] for plate_data in cropped_plates]
            
            if cropped_plates or detections:
                download_files, _, _ = create_download_files(annotated_image, cropped_plates, detections)
                if download_files is None:
                    print("Warning: Could not create download files")
        except Exception as e:
            print(f"Error in license plate processing: {e}")
            cropped_plates = []
            license_plate_gallery = []
            download_files = None
    
    # Create stats text
    stats_text = ""
    if show_stats and detections:
        df = pd.DataFrame(detections)
        counts = df['Object'].value_counts().to_dict()
        stats_text = "Detection Summary:\n"
        for obj, count in counts.items():
            stats_text += f"- {obj}: {count}\n"
        
        if cropped_plates:
            stats_text += f"\nLicense Plates Cropped: {len(cropped_plates)}\n"
    
    # Add stats to image if requested
    if show_stats and stats_text:
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), stats_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([10, 10, 20 + text_width, 20 + text_height], fill=(0, 0, 0, 128))
        
        draw.text((15, 15), stats_text, font=font, fill=(255, 255, 255))
    
    detection_table = pd.DataFrame(detections) if detections else pd.DataFrame(columns=["Object", "Confidence", "Position", "Dimensions"])
    
    return annotated_image, detection_table, stats_text, license_plate_gallery, download_files

# Define custom CSS for styling
custom_css = """
#title { text-align: center; }
#description { text-align: center; }
.footer { 
    text-align: center; 
    margin-top: 20px;
    color: #666;
}
.important { font-weight: bold; color: red; }
.download-section {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}
"""

# Set up Gradio interface with Blocks for more control
with gr.Blocks(css=custom_css, title="YOLOv11 Motorcyclist Helmet Detection") as demo:
    gr.HTML("<h1 id='title'>YOLOv11 Motorcyclist Helmet Detection with License Plate Extraction</h1>")
    gr.HTML("""
    <div id='description'>
        <p>This application uses YOLOv11 to detect Motorcyclists with and without Helmets in images.</p>
        <p>Upload an image, adjust the parameters, and view the detection results with detailed statistics.</p>
        <p><strong>New Features:</strong> License plate cropping and downloadable results!</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Parameters")
            input_image = gr.Image(type="filepath", label="Input Image", sources=["upload", "webcam"])
            with gr.Row():
                image_size = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size")
                conf_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Confidence Threshold")
            with gr.Row():
                iou_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="IOU Threshold")
                show_stats = gr.Checkbox(value=True, label="Show Statistics on Image")
            
            crop_plates = gr.Checkbox(value=True, label="Crop License Plates")
            
            submit_btn = gr.Button("Detect Objects", variant="primary")
            clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=2):
            gr.Markdown("### Output Results")
            output_image = gr.Image(type="pil", label="Annotated Image")
            output_table = gr.Dataframe(
                headers=["Object", "Confidence", "Position", "Dimensions"],
                label="Detection Details",
                interactive=False
            )
            output_stats = gr.Textbox(label="Detection Summary", interactive=False)
            
            # License plate gallery
            gr.Markdown("### Cropped License Plates")
            license_gallery = gr.Gallery(
                label="Extracted License Plates",
                show_label=True,
                elem_id="license_gallery",
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto"
            )
            
            # Download section
            gr.Markdown("### Download Results")
            with gr.Group(elem_classes="download-section"):
                download_file = gr.File(
                    label="Download Complete Results (ZIP)",
                    interactive=False,
                    visible=True
                )
                gr.Markdown("*The ZIP file contains: annotated image, cropped license plates, and detection report (CSV)*")
    
    # Examples
    gr.Markdown("### Example Images")
    gr.Examples(
        examples=[["sample_1.jpg"], ["sample_2.jpg"], ["sample_3.jpg"], ["sample_4.jpg"], ["sample_5.jpg"]],
        inputs=input_image,
        outputs=[output_image, output_table, output_stats, license_gallery, download_file],
        fn=yoloV8_func,
        cache_examples=True,
    )
    
    # Footer
    gr.HTML("""
    <div class='footer'>
        <p>Built with Gradio and Ultralytics YOLO</p>
        <p>Note: This is a demonstration application. Detection accuracy may vary based on image quality and conditions.</p>
        <p><strong>License Plate Privacy:</strong> Extracted license plates are for demonstration purposes only.</p>
    </div>
    """)
    
    # Button actions
    submit_btn.click(
        fn=yoloV8_func,
        inputs=[input_image, image_size, conf_threshold, iou_threshold, show_stats, gr.State(True), crop_plates],
        outputs=[output_image, output_table, output_stats, license_gallery, download_file]
    )
    
    clear_btn.click(
        fn=lambda: [None, None, None, None, None],
        inputs=[],
        outputs=[input_image, output_image, output_table, output_stats, license_gallery, download_file]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)