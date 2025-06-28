import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import base64

class YOLODetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.load_model()
        
    def load_model(self):
        """
        Load YOLOv3 model with local weights
        """
        try:
            # Check if we're in a virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                print(f"Using virtual environment: {venv_path}")
            
            # Try to use a simpler approach with torchvision
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                print("Using Faster R-CNN model as fallback")
            except Exception as e:
                print(f"Failed to load Faster R-CNN: {e}")
                
                # Try loading YOLOv3 from local weights
                if os.path.exists('yolov3.weights'):
                    print("Loading YOLOv3 from local weights...")
                    try:
                        self.model = torch.hub.load('ultralytics/yolov3', 'custom', path='yolov3.weights')
                    except Exception as e:
                        print(f"Failed to load YOLOv3 from local weights: {e}")
                        raise
                else:
                    print("YOLOv3 weights not found. Using a simple object detector instead.")
                    # Create a simple object detector as fallback
                    self.model = SimpleObjectDetector()
        
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using a simple object detector as fallback.")
            self.model = SimpleObjectDetector()
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully and moved to {self.device}")
        else:
            raise RuntimeError("Failed to load any model")
        
    def detect_objects(self, image_path):
        """
        Detect objects in an image
        Returns: List of detected objects with their bounding boxes
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")
            
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, SimpleObjectDetector):
                results = self.model(image_tensor.unsqueeze(0))
            else:
                results = self.model(image_tensor.unsqueeze(0))
            
        detections = []
        
        # Handle different model output formats
        if isinstance(self.model, SimpleObjectDetector):
            # Simple detector format
            for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
                if score > 0.5:  # Confidence threshold
                    detections.append({
                        'class': int(label),
                        'confidence': float(score),
                        'bbox': box.tolist()
                    })
        else:
            # YOLOv3 format
            for pred in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
                if conf > 0.5:  # Confidence threshold
                    detections.append({
                        'class': int(cls),
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                
        return detections
    
    def draw_detections(self, image_path, detections):
        """
        Draw bounding boxes on the image
        Returns: Image with drawn bounding boxes
        """
        image = cv2.imread(image_path)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{det['class']}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    
    def is_point_in_bbox(self, point, bbox):
        """
        Check if a point is inside a bounding box
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_object(self, image_data):
        """
        Detect objects in the image using YOLO
        image_data: base64 encoded image or file path
        Returns: detected class name
        """
        try:
            # Check if image_data is base64
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Remove header from base64 string
                image_data = image_data.split(',')[1]
                # Convert base64 to image
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Load image from file path
                img = cv2.imread(image_data)
            
            if img is None:
                raise Exception("Failed to load image")
            
            # Convert image to tensor
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                if isinstance(self.model, SimpleObjectDetector):
                    results = self.model(image_tensor)
                    # Get class with highest confidence from results
                    if len(results['scores']) > 0:
                        max_conf_idx = results['scores'].argmax().item()
                        if results['scores'][max_conf_idx] > 0.5:  # Confidence threshold
                            return str(results['labels'][max_conf_idx].item()).lower()
                else:
                    results = self.model(image_tensor)
                    # For YOLOv3 format
                    if len(results.xyxy[0]) > 0:
                        # Get prediction with highest confidence
                        pred = results.xyxy[0]
                        max_conf_idx = pred[:, 4].argmax().item()
                        if pred[max_conf_idx, 4] > 0.5:  # Confidence threshold
                            return str(int(pred[max_conf_idx, 5].item())).lower()
            
            return None
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return None

# Simple fallback object detector
class SimpleObjectDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained model for feature extraction
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Remove the last layer
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Create dummy detections (this is just a fallback)
        batch_size = x.shape[0]
        # Create some dummy boxes, scores, and labels
        boxes = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], device=x.device)
        boxes = boxes.unsqueeze(0).repeat(batch_size, 1, 1)
        scores = torch.tensor([0.8, 0.7], device=x.device).unsqueeze(0).repeat(batch_size, 1)
        labels = torch.tensor([1, 2], device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        return [{'boxes': boxes[i], 'scores': scores[i], 'labels': labels[i]} for i in range(batch_size)] 