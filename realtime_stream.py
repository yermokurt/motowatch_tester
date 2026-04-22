import cv2
import torch
import threading
import queue
import time
from ultralytics import YOLO
from detector import label_map
from license_plate_ocr import extract_license_plate_text

# ===== CONFIGURATION =====
CAMERA_SOURCE = 0  # 0 for webcam, or use "rtsp://..." for IP camera
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# ===== LOAD MODELS (Optimized for GTX 1650 Ti) =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

helmet_model = YOLO("helmet_best_v2.pt").to(device)
plate_model = YOLO("license_best.pt").to(device)
moto_model = YOLO("best.pt").to(device)

# Buffer to hold the latest frame
frame_queue = queue.Queue(maxsize=1)

def capture_thread(source):
    """Constantly reads frames from the camera to ensure zero-latency."""
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Drop old frame and put new one to keep it 'Live'
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except: pass
    cap.release()

def main():
    # Start capture thread
    threading.Thread(target=capture_thread, args=(CAMERA_SOURCE,), daemon=True).start()
    
    print("Starting Real-Time Inference... Press 'q' to quit.")
    
    while True:
        if frame_queue.empty(): continue
        
        frame = frame_queue.get()
        display_frame = frame.copy()
        
        # 1. Run Triple-Model Ensemble (FP16 Half Mode for Speed)
        h_res = helmet_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, half=True, verbose=False)
        p_res = plate_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, half=True, verbose=False)
        m_res = moto_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, half=True, verbose=False)
        
        has_violation = False
        
        # 2. Process Helmet Detections
        if h_res[0].boxes:
            for box, cls in zip(h_res[0].boxes.xyxy, h_res[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                raw_label = helmet_model.names[int(cls)]
                label = label_map.get(raw_label, raw_label)
                color = (0, 255, 0) if label == "Compliant" else (0, 0, 255)
                if label == "Non-compliant": has_violation = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Process Plate Detections
        if p_res[0].boxes:
            for box in p_res[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Trigger OCR only on Violation
                if has_violation:
                    # Pad crop
                    pw, ph = (x2-x1)*0.1, (y2-y1)*0.1
                    px1, py1 = max(0, int(x1-pw)), max(0, int(y1-ph))
                    px2, py2 = int(x2+pw), int(y2+ph)
                    crop = frame[py1:py2, px1:px2]
                    
                    # Async OCR would be better, but for this demo we'll do it inline
                    text = extract_license_plate_text(crop)
                    cv2.putText(display_frame, f"PLATE: {text}", (x1, y2+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Process Motorcycle Detections
        if m_res[0].boxes:
            for box, cls in zip(m_res[0].boxes.xyxy, m_res[0].boxes.cls):
                if int(cls) == 2: # motorcyclist
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        # Show Output
        cv2.imshow("Real-Time Helmet & Plate Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
