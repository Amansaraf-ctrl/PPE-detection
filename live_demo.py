import cv2
import json
from ultralytics import YOLO

def main():
    print("Loading Binary Zero-Shot Engine...")
    model = YOLO("yolov8s-worldv2.pt")
    
    # We ask the AI to look for specific visual textures...
    custom_classes = [
        "person", 
        "heavy leather work boots",  # AI Class 1
        "casual sneakers",           # AI Class 2
        "open toe slippers",         # AI Class 3
        "bare feet"                  # AI Class 4
    ]
    model.set_classes(custom_classes)
    
    cap = cv2.VideoCapture(0)
    print("Webcam Active. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        results = model(frame, imgsz=640, conf=0.15, verbose=False)
        dashboard_payload = {"workers": [], "detected_footwear": []}
        
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- THE BINARY MAPPING LOGIC ---
            if class_name == "person":
                # Keep person neutral
                display_label = "Person"
                color = (255, 255, 255) # White
                
            elif class_name == "heavy leather work boots":
                # Map to Class 1: Safety Shoes
                display_label = "safety_shoes"
                color = (0, 255, 0) # Green
                dashboard_payload["detected_footwear"].append({"class": display_label, "bbox": [x1, y1, x2, y2]})
                
            elif class_name in ["casual sneakers", "open toe slippers", "bare feet"]:
                # Map everything else to Class 2: No Safety Shoes
                display_label = "no_safety_shoes"
                color = (0, 0, 255) # Red
                dashboard_payload["detected_footwear"].append({"class": display_label, "bbox": [x1, y1, x2, y2]})
                
            else:
                continue

            # Draw the UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Print JSON payload for the dashboard team
        if dashboard_payload["detected_footwear"]:
            print(json.dumps(dashboard_payload))

        cv2.imshow("Binary PPE Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()