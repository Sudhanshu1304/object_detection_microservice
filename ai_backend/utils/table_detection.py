import numpy as np
from typing import Optional
from utils.schema import DetectionResult
from ultralytics import YOLO

class TableDetector:
    """
    A class for detecting tables in document images using YOLO models.
    
    Attributes:
        model_path (Path): Path to the YOLO model weights
        confidence (float): Confidence threshold for detection
        iou_threshold (float): IoU threshold for NMS
    """
    
    def __init__(
        self,
        confidence: float = 0.50,
        iou_threshold: float = 0.3
    ) -> None:
        """
        Initialize the TableDetector with model and parameters.
        
        Args:
            model_path: Path to the YOLO model weights
            confidence: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = 'ai_backend/model/table-detection-and-extraction.pt'
        self.model = YOLO(str(self.model_path))
        self.min_conf = confidence
        self.iou = iou_threshold

    def detect(self, image: np.array) -> Optional[np.ndarray]:
        """
        Detect tables in the given image.
        
        Args:
            image: Input image
            
        Returns:
            Array of bounding box coordinates or None if no tables detected
        """
        results = self.model.predict(image, verbose=False, iou = self.iou, conf = self.min_conf)
        detections = []
        # Process the results from the model
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Append the structured detection result
                detections.append(
                    DetectionResult(
                        class_name=class_name,
                        confidence=round(confidence, 4),
                        bounding_box=[x1, y1, x2, y2]
                    )
                )
        return detections