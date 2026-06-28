import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Resolve import paths: support running from project root or inside the Source directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt imports using different folder naming conventions
try:
    from src.models import DeepIrisResNet
except ModuleNotFoundError:
    try:
        from Source.models import DeepIrisResNet
    except ModuleNotFoundError:
        try:
            from models import DeepIrisResNet
        except ModuleNotFoundError:
            # Fallback inline model definition if imports fail completely
            class DeepIrisResNet(nn.Module):
                def __init__(self, num_classes=224):
                    super().__init__()
                    from torchvision.models import resnet50
                    self.model = resnet50(pretrained=True)
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, num_classes)
                def forward(self, x):
                    return self.model(x)

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Please run: pip install opencv-python")
    sys.exit(1)

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    num_classes = 224
    model = DeepIrisResNet(num_classes=num_classes)
    
    # Path to the trained checkpoint
    checkpoint_path = os.path.join(project_root, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(current_dir, "checkpoints", "best_model.pth")
        
    model_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading weights from: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model_loaded = True
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Running in DEMO MODE with uninitialized weights.")
    else:
        print(f"Warning: Checkpoint not found at '{checkpoint_path}'.")
        print("Please train the model first using training.py or download the checkpoints.")
        print("Running in DEMO MODE with random/uninitialized weights (outputs will be random).")

    model.to(device)
    model.eval()

    # Preprocessing pipeline matching DeepIris training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Try loading eye Haar Cascade for auto-detection
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    auto_detect = not eye_cascade.empty()
    if auto_detect:
        print("Haar Cascade eye detector loaded successfully.")
    else:
        print("Warning: Haar Cascade eye detector could not be loaded. Defaulting to manual alignment ROI.")

    # Start Webcam
    print("Starting webcam... Press 'q' to quit, 'm' to toggle Auto-Detection vs Manual ROI.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    mode = "manual"  # modes: 'manual' or 'auto'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()
        
        # Default Prediction Variables
        pred_class = "N/A"
        confidence = 0.0
        eye_crop = None
        roi_coords = None

        if mode == "auto" and auto_detect:
            # Convert frame to grayscale for cascade classifier
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))
            
            # If multiple eyes are found, pick the largest one (likely closest to camera)
            if len(eyes) > 0:
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
                ex, ey, ew, eh = eyes[0]
                roi_coords = (ex, ey, ew, eh)
                
                # Crop eye region with a small padding
                pad = int(min(ew, eh) * 0.1)
                y1 = max(0, ey - pad)
                y2 = min(h, ey + eh + pad)
                x1 = max(0, ex - pad)
                x2 = min(w, ex + ew + pad)
                eye_crop = frame[y1:y2, x1:x2]
        
        else:
            # Manual mode: Use a static central Box of size 160x160 as the region of interest (ROI)
            roi_size = 160
            cx, cy = w // 2, h // 2
            x1, y1 = cx - roi_size // 2, cy - roi_size // 2
            x2, y2 = cx + roi_size // 2, cy + roi_size // 2
            roi_coords = (x1, y1, roi_size, roi_size)
            eye_crop = frame[y1:y2, x1:x2]

        # Draw Guide ROI Box
        if roi_coords is not None:
            rx, ry, rw, rh = roi_coords
            # Color changes to green if model is loaded, yellow if not
            box_color = (0, 255, 0) if model_loaded else (0, 255, 255)
            cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            if mode == "manual":
                cv2.putText(display_frame, "Align eye inside box", (rx, ry - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "Detected Eye", (rx, ry - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # Run inference if we have a valid cropped eye region
        if eye_crop is not None and eye_crop.size > 0:
            try:
                # Preprocess cropped image to match dataset processing
                # Convert BGR to Grayscale PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY))
                
                # Apply same transform pipeline (Resize, ToTensor, Channel replication, Normalize)
                tensor_img = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probabilities = torch.softmax(outputs, dim=1)
                    conf, preds = torch.max(probabilities, dim=1)
                    
                    pred_class = f"Subject_{preds.item() + 1:03d}"
                    confidence = conf.item() * 100
            except Exception as e:
                # Silently catch preprocessing errors (e.g. empty crop dimensions)
                pass

        # Overlay results on the live output window
        # Create a status bar at the top
        cv2.rectangle(display_frame, (0, 0), (w, 45), (30, 30, 30), -1)
        
        status_text = f"Mode: {mode.upper()} | Model Loaded: {str(model_loaded).upper()}"
        cv2.putText(display_frame, status_text, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        if eye_crop is not None:
            pred_text = f"Pred: {pred_class} (Conf: {confidence:.1f}%)"
            text_color = (0, 255, 0) if confidence > 70.0 else (0, 165, 255)
            if not model_loaded:
                pred_text += " [Demo Mode]"
                text_color = (0, 255, 255)
            cv2.putText(display_frame, pred_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

        cv2.imshow("Iris Recognition Live Demo", display_frame)

        # Handle Keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            if mode == "manual" and auto_detect:
                mode = "auto"
                print("Switched to AUTO (Eye-detection) mode.")
            else:
                mode = "manual"
                print("Switched to MANUAL ROI mode.")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Demo finished.")

if __name__ == "__main__":
    main()
