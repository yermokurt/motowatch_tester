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

# ===== Optional OCR imports =====
try:
    from license_plate_ocr import extract_license_plate_text
    OCR_AVAILABLE = True
    print("Basic OCR module loaded successfully")
except ImportError as e:
    print(f"Basic OCR module not available: {e}")
    OCR_AVAILABLE = False

try:
    from advanced_ocr import (
        extract_license_plate_text_advanced,
        get_available_models,
        set_ocr_model,
    )
    ADVANCED_OCR_AVAILABLE = True
    print("Advanced OCR module loaded successfully")
except ImportError as e:
    print(f"Advanced OCR module not available: {e}")
    ADVANCED_OCR_AVAILABLE = False

# ===== Sample images (optional) =====
torch.hub.download_url_to_file(
    "https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-1.jpg?raw=true",
    "sample_1.jpg",
)
torch.hub.download_url_to_file(
    "https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-2.jpg?raw=true",
    "sample_2.jpg",
)
torch.hub.download_url_to_file(
    "https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-3.jpg?raw=true",
    "sample_3.jpg",
)
torch.hub.download_url_to_file(
    "https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-4.jpg?raw=true",
    "sample_4.jpg",
)
torch.hub.download_url_to_file(
    "https://github.com/Janno1402/Helmet-License-Plate-Detection/blob/main/Sample-Image-5.jpg?raw=true",
    "sample_5.jpg",
)

# ===== Model & class names =====
model = YOLO("best.pt")  # make sure best.pt is present
class_names = {0: "With Helmet", 1: "Without Helmet", 2: "License Plate"}


# ===== Helpers =====
def crop_license_plates(image, detections, extract_text=False, selected_ocr_model="auto"):
    """Crop license plates and (optionally) run OCR on the crops."""
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
            if detection["Object"] != "License Plate":
                continue

            pos_str = detection["Position"].strip("()")
            if "," not in pos_str:
                print(
                    f"Error: Invalid position format for detection {i}: {detection['Position']}"
                )
                continue

            x1, y1 = map(int, pos_str.split(", "))

            dims_str = detection["Dimensions"]
            if "x" not in dims_str:
                print(
                    f"Error: Invalid dimensions format for detection {i}: {detection['Dimensions']}"
                )
                continue

            width, height = map(int, dims_str.split("x"))

            if width <= 0 or height <= 0:
                print(f"Error: Invalid dimensions for detection {i}: {width}x{height}")
                continue

            x2, y2 = x1 + width, y1 + height

            if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                print(
                    f"Warning: Bounding box extends beyond image boundaries for detection {i}"
                )
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                print(
                    f"Error: Invalid crop coordinates for detection {i}: ({x1},{y1}) to ({x2},{y2})"
                )
                continue

            cropped_plate = image.crop((x1, y1, x2, y2))

            if cropped_plate.size[0] == 0 or cropped_plate.size[1] == 0:
                print(
                    f"Error: Cropped image has zero dimensions for detection {i}"
                )
                continue

            plate_data = {
                "image": cropped_plate,
                "confidence": detection["Confidence"],
                "position": detection["Position"],
                "crop_coords": f"({x1},{y1}) to ({x2},{y2})",
                "text": "Processing...",
            }

            if extract_text and (OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE):
                try:
                    print(
                        f"Extracting text from license plate {i+1} using {selected_ocr_model}..."
                    )

                    if ADVANCED_OCR_AVAILABLE and selected_ocr_model != "basic":
                        if selected_ocr_model != "auto":
                            set_ocr_model(selected_ocr_model)
                        plate_text = extract_license_plate_text_advanced(
                            cropped_plate,
                            None if selected_ocr_model == "auto" else selected_ocr_model,
                        )
                    else:
                        plate_text = extract_license_plate_text(cropped_plate)

                    if (
                        plate_text
                        and plate_text.strip()
                        and not plate_text.startswith("Error")
                    ):
                        plate_data["text"] = plate_text.strip()
                        print(f"Extracted text: {plate_text.strip()}")
                    else:
                        plate_data["text"] = "No text detected"
                        print(f"No text found in plate {i+1}")
                except Exception as e:
                    print(f"OCR extraction failed for plate {i+1}: {e}")
                    plate_data["text"] = f"OCR Failed: {str(e)}"
            elif extract_text and not (OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE):
                plate_data["text"] = "OCR not available"
            else:
                plate_data["text"] = "OCR disabled"

            cropped_plates.append(plate_data)

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
                plate_data["image"].save(plate_path, quality=95)
                plate_paths.append(plate_path)
            except Exception as e:
                print(f"Error saving license plate {i+1}: {e}")
                continue

        report_data = []
        for detection in detections:
            report_data.append(detection)

        for i, plate_data in enumerate(cropped_plates):
            report_data.append(
                {
                    "Object": f"License Plate {i+1} - Text",
                    "Confidence": plate_data["confidence"],
                    "Position": plate_data["position"],
                    "Dimensions": "Extracted Text",
                    "Text": plate_data.get("text", "N/A"),
                }
            )

        report_path = f"temp/detection_report_{timestamp}.csv"
        if report_data:
            try:
                df = pd.DataFrame(report_data)
                df.to_csv(report_path, index=False)
            except Exception as e:
                print(f"Error creating detection report: {e}")
                report_path = None

        zip_path = f"temp/detection_results_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
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


# ===== Main inference =====
def yoloV8_func(
    image=None,
    image_size=640,
    conf_threshold=0.4,
    iou_threshold=0.5,
    show_stats=True,
    show_confidence=True,
    crop_plates=True,
    extract_text=False,
    ocr_on_no_helmet=False,
    selected_ocr_model="auto",
):
    if image_size is None:
        image_size = 640
    if not isinstance(image_size, int):
        image_size = int(image_size)

    imgsz = [image_size, image_size]
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz)

    annotated_image = results[0].plot()
    if isinstance(annotated_image, np.ndarray):
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    boxes = results[0].boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls)
            confidence = float(conf)
            label = class_names.get(class_id, f"Class {class_id}")
            detections.append(
                {
                    "Object": label,
                    "Confidence": f"{confidence:.2f}",
                    "Position": f"({int(x1)}, {int(y1)})",
                    "Dimensions": f"{int(x2 - x1)}x{int(y2 - y1)}",
                }
            )

    cropped_plates = []
    license_plate_gallery = []
    plate_texts = []
    download_files = None

    has_no_helmet = any(d["Object"] == "Without Helmet" for d in detections)
    should_extract_text = extract_text or (ocr_on_no_helmet and has_no_helmet)
    ocr_available = OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE

    if crop_plates and detections:
        try:
            license_plate_count = len([d for d in detections if d["Object"] == "License Plate"])
            print(f"Processing {license_plate_count} license plates...")

            if ocr_on_no_helmet and has_no_helmet:
                print("⚠️  No helmet detected - OCR will be performed on license plates")

            cropped_plates = crop_license_plates(
                image, detections, should_extract_text, selected_ocr_model
            )
            print(f"Successfully cropped {len(cropped_plates)} license plates")

            license_plate_gallery = [plate_data["image"] for plate_data in cropped_plates]

            if should_extract_text and ocr_available:
                print("Extracting text from license plates...")
                plate_texts = []
                for i, plate_data in enumerate(cropped_plates):
                    text = plate_data.get("text", "No text detected")
                    print(f"Plate {i+1} text: {text}")
                    if ocr_on_no_helmet and has_no_helmet:
                        plate_texts.append(f"🚨 No Helmet Violation - Plate {i+1}: {text}")
                    else:
                        plate_texts.append(f"Plate {i+1}: {text}")
            elif should_extract_text and not ocr_available:
                plate_texts = [
                    "OCR not available - install requirements: pip install transformers easyocr"
                ]
            elif not should_extract_text:
                plate_texts = [
                    f"Plate {i+1}: Text extraction disabled" for i in range(len(cropped_plates))
                ]

            if cropped_plates or detections:
                download_files, _, _ = create_download_files(
                    annotated_image, cropped_plates, detections
                )
                if download_files is None:
                    print("Warning: Could not create download files")
        except Exception as e:
            print(f"Error in license plate processing: {e}")
            cropped_plates = []
            license_plate_gallery = []
            plate_texts = ["Error processing license plates"]
            download_files = None

    stats_text = ""
    if show_stats and detections:
        df = pd.DataFrame(detections)
        counts = df["Object"].value_counts().to_dict()
        stats_text = "Detection Summary:\n"
        for obj, count in counts.items():
            stats_text += f"- {obj}: {count}\n"

        if cropped_plates:
            stats_text += f"\nLicense Plates Cropped: {len(cropped_plates)}\n"
            if has_no_helmet:
                stats_text += "⚠️ HELMET VIOLATION DETECTED!\n"
            if should_extract_text and (OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE):
                stats_text += "Extracted Text:\n"
                for i, plate_data in enumerate(cropped_plates):
                    text = plate_data.get("text", "No text")
                    if has_no_helmet and ocr_on_no_helmet:
                        stats_text += f"🚨 Violation - Plate {i+1}: {text}\n"
                    else:
                        stats_text += f"- Plate {i+1}: {text}\n"

    detection_table = (
        pd.DataFrame(detections)
        if detections
        else pd.DataFrame(columns=["Object", "Confidence", "Position", "Dimensions"])
    )
    plate_text_output = (
        "\n".join(plate_texts)
        if plate_texts
        else "No license plates detected or OCR disabled"
    )

    return (
        annotated_image,
        detection_table,
        stats_text,
        license_plate_gallery,
        download_files,
        plate_text_output,
    )


# ====== UI ======
custom_css = """
#title { text-align: center; }
#description { text-align: center; }
.footer { text-align: center; margin-top: 20px; color: #666; }
.important { font-weight: bold; color: red; }
.download-section { background-color: #f6f6f6; padding: 15px; border-radius: 10px; margin-top: 10px; }
.ocr-section { background-color: #eef7ff; padding: 15px; border-radius: 10px; margin-top: 10px; }
.card { background: white; border-radius: 16px; padding: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); }
"""

def toggle_sections(extract_text_checked, crop_checked):
    """Control visibility of Cropped Plates & OCR sections.
    Requirement: If user checks the OCR checkbox, show the cropped plates AND OCR text.
    Otherwise, hide both sections. Crops also require crop_checkbox True.
    """
    show_gallery = bool(extract_text_checked and crop_checked)
    show_ocr = bool(extract_text_checked)
    return (
        gr.update(visible=show_gallery),  # license_gallery
        gr.update(visible=show_ocr),      # ocr group container (textbox)
    )


with gr.Blocks(
    css=custom_css,
    title="YOLOv11 Motorcyclist Helmet Detection",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.HTML("<h1 id='title'>YOLOv11 Motorcyclist Helmet Detection</h1>")
    gr.HTML(
        f"""
    <div id='description'>
        <p>This app detects motorcyclists <strong>with</strong> / <strong>without</strong> helmets and can optionally read license plates.</p>
        <p><strong>OCR Status:</strong>
        {'✅ Advanced OCR Available' if ADVANCED_OCR_AVAILABLE else '🟡 Basic OCR Available' if OCR_AVAILABLE else '❌ OCR Not Available (install requirements)'}
        </p>
    </div>
    """
    )

    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Input Parameters")
                        input_image = gr.Image(
                            type="filepath", label="Input Image", sources=["upload", "webcam"]
                        )
                        with gr.Row():
                            image_size = gr.Slider(
                                minimum=320, maximum=1280, value=640, step=32, label="Image Size"
                            )
                            conf_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Confidence Threshold"
                            )
                        with gr.Row():
                            iou_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="IOU Threshold"
                            )
                            show_stats = gr.Checkbox(value=True, label="Show Stats on Image")

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Options")
                        crop_plates = gr.Checkbox(value=True, label="Enable License Plate Cropping")

                        ocr_available = OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE
                        if ocr_available:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="Enable OCR (Show Cropped Plates & Text)",
                                info="When enabled: shows cropped plates + runs OCR",
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=True,
                                label="🚨 Auto-OCR when No Helmet Detected",
                            )

                            if ADVANCED_OCR_AVAILABLE:
                                models = get_available_models()
                                model_choices = [("Auto (Recommended)", "auto"), ("Basic EasyOCR", "basic")]
                                for key, info in models.items():
                                    model_choices.append((info["name"], key))
                                selected_ocr_model = gr.Dropdown(
                                    choices=model_choices,
                                    value="auto",
                                    label="OCR Model Selection",
                                    info="Choose OCR model (Advanced models may require setup)",
                                )
                            else:
                                selected_ocr_model = gr.State("basic")

                            gr.Markdown("*Note: OCR may increase processing time*")
                        else:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="OCR Not Available",
                                interactive=False,
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=False,
                                label="🚨 Auto-OCR when No Helmet (Not Available)",
                                interactive=False,
                            )
                            selected_ocr_model = gr.State("basic")

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Detect", variant="primary")
                            clear_btn = gr.Button("🧹 Clear")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Output")
                        output_image = gr.Image(type="pil", label="Annotated Image")
                        output_table = gr.Dataframe(
                            headers=["Object", "Confidence", "Position", "Dimensions"],
                            label="Detection Details",
                            interactive=False,
                        )
                        output_stats = gr.Textbox(
                            label="Detection Summary", interactive=False, lines=6
                        )

                    # ---- Cropped plates & OCR (conditionally visible) ----
                    license_gallery = gr.Gallery(
                        label="Extracted License Plates",
                        show_label=True,
                        elem_id="license_gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                        height="auto",
                        visible=False,  # hidden until OCR checkbox is enabled
                    )

                    ocr_group = gr.Group(elem_classes=["ocr-section"], visible=False)
                    with ocr_group:
                        gr.Markdown("### License Plate Text Recognition")
                        plate_text_output = gr.Textbox(
                            label="Extracted Text",
                            placeholder="License plate text will appear here when OCR is enabled",
                            lines=4,
                            interactive=False,
                        )

                    with gr.Group(elem_classes=["download-section", "card"]):
                        gr.Markdown("### Download Results")
                        download_file = gr.File(
                            label="Download Complete Results (ZIP)",
                            interactive=False,
                            visible=True,
                        )
                        gr.Markdown(
                            "*ZIP contains: annotated image, cropped plates (if any), and a CSV report with OCR results*"
                        )

        with gr.TabItem("Examples"):
            gr.Markdown("### Example Images")
            gr.Examples(
                examples=[["sample_1.jpg"], ["sample_2.jpg"], ["sample_3.jpg"], ["sample_4.jpg"], ["sample_5.jpg"]],
                inputs=input_image,
                outputs=[
                    output_image,
                    output_table,
                    output_stats,
                    license_gallery,
                    download_file,
                    plate_text_output,
                ],
                fn=lambda img: yoloV8_func(
                    img, 640, 0.4, 0.5, True, True, True, False
                ),
                cache_examples=True,
            )

    gr.HTML(
        """
    <div class='footer'>
        <p>Built with Gradio and Ultralytics YOLO</p>
        <p><strong>License Plate Privacy:</strong> Extracted plate images & text are for demo purposes only.</p>
        <p><strong>Requirements for OCR:</strong> torch, transformers, easyocr, opencv-python</p>
    </div>
    """
    )

    # ===== Wire events =====
    # 1) Main click
    submit_btn.click(
        fn=yoloV8_func,
        inputs=[
            input_image,
            image_size,
            conf_threshold,
            iou_threshold,
            show_stats,
            gr.State(True),  # show_confidence placeholder
            crop_plates,
            extract_text,
            ocr_on_no_helmet,
            selected_ocr_model,
        ],
        outputs=[
            output_image,
            output_table,
            output_stats,
            license_gallery,
            download_file,
            plate_text_output,
        ],
    )

    # 2) Clear
    clear_btn.click(
        fn=lambda: [None, None, None, None, None, None],
        inputs=[],
        outputs=[
            input_image,
            output_image,
            output_table,
            output_stats,
            license_gallery,
            download_file,
            plate_text_output,
        ],
    )

    # 3) Toggle visibility when user toggles OCR or Crop checkboxes
    extract_text.change(
        fn=toggle_sections,
        inputs=[extract_text, crop_plates],
        outputs=[license_gallery, ocr_group],
    )
    crop_plates.change(
        fn=toggle_sections,
        inputs=[extract_text, crop_plates],
        outputs=[license_gallery, ocr_group],
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
