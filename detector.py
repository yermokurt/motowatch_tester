import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
import zipfile

# ===== LOAD TWO MODELS =====
helmet_model = YOLO("helmet_best_v2.pt")   # your helmet model
plate_model = YOLO("license_best.pt")      # license plate model
motorcycle_model = YOLO("best.pt")         # for motorcycle detection

# No hardcoded class names needed, we use model.names

label_map = {
    "valid_helmet": "Compliant",
    "invalid_helmet": "Non-compliant"
}

# ===== OCR (unchanged) =====
try:
    from license_plate_ocr import extract_license_plate_text
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

try:
    from advanced_ocr import (
        extract_license_plate_text_advanced,
        set_ocr_model,
    )
    ADVANCED_OCR_AVAILABLE = True
except:
    ADVANCED_OCR_AVAILABLE = False

# ===== MAIN DETECTION =====
def yolov8_detect(
    image=None,
    image_size=640,
    conf_threshold=0.3,
    iou_threshold=0.3,
    show_stats=True,
    show_confidence=True,
    crop_plates=True,
    extract_text=False,
    ocr_on_no_helmet=False,
    selected_ocr_model="auto",
):

    # ===== FIX INPUT (GRADIO) =====
    if isinstance(image, dict) and "image" in image:
        image = image["image"]

    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, str):
        # If it's a path, load it as a numpy array (BGR)
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not load image from path: {image}")

    if image is None:
        raise ValueError("No image provided")

    imgsz = [image_size, image_size]

    # ===== RUN BOTH MODELS =====
    helmet_results = helmet_model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, half=True)
    plate_results = plate_model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, half=True)
    moto_results = motorcycle_model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, half=True)

    detections = []

    # ===== HELMET DETECTION =====
    if helmet_results[0].boxes is not None:
        for box, cls, conf in zip(
            helmet_results[0].boxes.xyxy,
            helmet_results[0].boxes.cls,
            helmet_results[0].boxes.conf,
        ):
            x1, y1, x2, y2 = box.tolist()
            raw_label = helmet_model.names[int(cls)]
            label = label_map.get(raw_label, raw_label)

            detections.append({
                "Object": label,
                "Confidence": f"{float(conf):.2f}",
                "Position": f"({int(x1)}, {int(y1)})",
                "Dimensions": f"{int(x2-x1)}x{int(y2-y1)}",
            })

    # ===== LICENSE PLATE / MOTORCYCLE DETECTION =====
    if plate_results[0].boxes is not None:
        for box, cls, conf in zip(
            plate_results[0].boxes.xyxy,
            plate_results[0].boxes.cls,
            plate_results[0].boxes.conf,
        ):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls)
            
            # Use model names for mapping
            # plate_model.names: {0: 'helmet', 1: 'licenseplate', 2: 'motorcyclist', 3: 'nohelmet'}
            class_id = int(cls)
            
            # license_best.pt classes: {0: 'license_plate'}
            if class_id == 0:
                label = "License Plate"
            else:
                continue

            detections.append({
                "Object": label,
                "Confidence": f"{float(conf):.2f}",
                "Position": f"({int(x1)}, {int(y1)})",
                "Dimensions": f"{int(x2-x1)}x{int(y2-y1)}",
            })

    # ===== MOTORCYCLE DETECTION (best.pt) =====
    if moto_results[0].boxes is not None:
        for box, cls, conf in zip(moto_results[0].boxes.xyxy, moto_results[0].boxes.cls, moto_results[0].boxes.conf):
            class_id = int(cls)
            if class_id == 2: # motorcyclist in best.pt
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "Object": "Motorcycle",
                    "Confidence": f"{float(conf):.2f}",
                    "Position": f"({int(x1)}, {int(y1)})",
                    "Dimensions": f"{int(x2-x1)}x{int(y2-y1)}",
                })

    # ===== DRAW OUTPUT IMAGE =====
    # We'll draw boxes manually to ensure correct labels and colors
    if isinstance(image, Image.Image):
        img_draw = np.array(image.convert("RGB"))
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    else:
        img_draw = image.copy()

    # Draw Helmet Detections
    if helmet_results[0].boxes is not None:
        for box, cls, conf_val in zip(helmet_results[0].boxes.xyxy, helmet_results[0].boxes.cls, helmet_results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            raw_label = helmet_model.names[int(cls)]
            label = label_map.get(raw_label, raw_label)
            color = (0, 255, 0) if label == "Compliant" else (0, 0, 255)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_draw, f"{label} {float(conf_val):.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw plate and motorcycle boxes
    if plate_results[0].boxes is not None:
        for box, cls in zip(plate_results[0].boxes.xyxy, plate_results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_id = int(cls)
            
            if class_id == 0: # License Plate
                color = (255, 0, 0)
                label = "License Plate"
            else:
                continue

            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Motorcycle boxes
    if moto_results[0].boxes is not None:
        for box, cls in zip(moto_results[0].boxes.xyxy, moto_results[0].boxes.cls):
            if int(cls) == 2: # motorcyclist
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(img_draw, "Motorcycle", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    img = img_draw

    # ===== OCR + CROPPING =====
    cropped_plates = []
    plate_texts = []

    has_no_helmet = any(d["Object"] == "Non-compliant" for d in detections)
    do_ocr = extract_text or (ocr_on_no_helmet and has_no_helmet)

    if plate_results[0].boxes is not None:
        for i, box in enumerate(plate_results[0].boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            # Add padding for better OCR (10% on each side)
            pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            px1, py1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            px2, py2 = int(x2 + pw), int(y2 + ph)

            pil_img = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
            crop = pil_img.crop((px1, py1, px2, py2))
            cropped_plates.append(crop)

            text = "OCR disabled"

            if do_ocr and (OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE):
                try:
                    if ADVANCED_OCR_AVAILABLE:
                        if selected_ocr_model != "auto":
                            set_ocr_model(selected_ocr_model)
                        text = extract_license_plate_text_advanced(crop)
                    else:
                        text = extract_license_plate_text(crop)
                except:
                    text = "OCR Failed"

            if has_no_helmet and ocr_on_no_helmet:
                plate_texts.append(f"🚨 Violation Plate {i+1}: {text}")
            else:
                plate_texts.append(f"Plate {i+1}: {text}")

    # ===== STATS =====
    stats = "No objects detected."
    if detections:
        df = pd.DataFrame(detections)
        if "Object" in df.columns:
            counts = df["Object"].value_counts().to_dict()
            stats = "Detection Summary:\n"
            for k, v in counts.items():
                stats += f"- {k}: {v}\n"
            
            if has_no_helmet:
                stats += "\n⚠️ HELMET VIOLATION DETECTED\n"
    
    detection_table = pd.DataFrame(detections) if detections else pd.DataFrame(columns=["Object", "Confidence", "Position", "Dimensions"])

    # ===== ZIP CREATION =====
    zip_path = None
    if crop_plates and cropped_plates:
        try:
            zip_filename = f"plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(os.getcwd(), zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for i, crop in enumerate(cropped_plates):
                    # Save PIL image to temporary buffer then to zip
                    crop_filename = f"plate_{i+1}.png"
                    crop.save(crop_filename)
                    zipf.write(crop_filename)
                    os.remove(crop_filename)
        except Exception as e:
            print(f"Zip creation error: {e}")

    # ===== FINAL RESULTS =====
    gallery_data = []
    for i, crop in enumerate(cropped_plates):
        # Match text to crop if possible, otherwise use generic label
        label = "No OCR"
        # We store raw texts in plate_texts, we need to extract the actual plate string
        # plate_texts looks like ["Plate 1: ABC1234", "🚨 Violation Plate 2: XYZ7890"]
        if i < len(plate_texts):
            label = plate_texts[i].split(": ")[-1]
        gallery_data.append((crop, label))

    try:
        annotated_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return (
            annotated_image,
            detection_table,
            stats,
            gallery_data,
            zip_path,
            "\n".join(plate_texts) if plate_texts else "No plates detected",
        )
    except Exception as e:
        import traceback
        print(f"Error in yolov8_detect: {e}")
        traceback.print_exc()
        return None, pd.DataFrame(), f"Error: {str(e)}", [], None, "Detection failed"

def yolov8_video_detect(
    video_path,
    image_size=640,
    conf_threshold=0.3,
    iou_threshold=0.3,
    frame_skip=2  # Process every Nth frame
):
    if not video_path:
        return None, "No video provided"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Error opening video at {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    # Try different codecs for maximum compatibility and robustness
    codecs = ['avc1', 'X264', 'mp4v', 'XVID']
    out = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if out.isOpened():
                print(f"Successfully initialized VideoWriter with codec: {codec}")
                break
            else:
                out = None
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            out = None

    if out is None:
        return None, "Error: Could not initialize any video codec on this system."

    imgsz = [image_size, image_size]
    frame_count = 0
    annotated_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only run detection every Nth frame
        if frame_count % frame_skip == 0:
            h_res = helmet_model.predict(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False, half=True)
            p_res = plate_model.predict(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False, half=True)
            m_res = motorcycle_model.predict(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False, half=True)

            annotated_frame = frame.copy()
            
            # Draw Helmet Detections (best1.pt)
            if h_res[0].boxes is not None:
                for box, cls, conf in zip(h_res[0].boxes.xyxy, h_res[0].boxes.cls, h_res[0].boxes.conf):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    raw_label = helmet_model.names[int(cls)]
                    label = label_map.get(raw_label, raw_label)
                    color = (0, 255, 0) if label == "Compliant" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw Plate Detections
            if p_res[0].boxes is not None:
                for box, cls, conf in zip(p_res[0].boxes.xyxy, p_res[0].boxes.cls, p_res[0].boxes.conf):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_id = int(cls)
                    if class_id == 0: # License Plate
                        color = (255, 0, 0)
                        label = "License Plate"
                    else:
                        continue
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw Motorcycle Detections
            if m_res[0].boxes is not None:
                for box, cls, conf in zip(m_res[0].boxes.xyxy, m_res[0].boxes.cls, m_res[0].boxes.conf):
                    if int(cls) == 2: # motorcyclist
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.putText(annotated_frame, f"Motorcycle {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # If we skipped this frame's detection, we write the raw frame or the last known annotated frame
        # To avoid "frozen" boxes, we'll write the raw frame for skipped frames
        if frame_count % frame_skip == 0:
            out.write(annotated_frame)
        else:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    return output_path, f"Processed {total_frames} frames (Skipped {frame_skip-1} frames between each detection)."

def download_sample_images():
    import torch
    # (Just keeping the placeholder comments clean this time)
    pass

def get_ocr_status():
    return {
        "basic_available": OCR_AVAILABLE,
        "advanced_available": ADVANCED_OCR_AVAILABLE,
        "any_available": OCR_AVAILABLE or ADVANCED_OCR_AVAILABLE
    }