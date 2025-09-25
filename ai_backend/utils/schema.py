from pydantic import BaseModel, Field
from typing import List



class DetectionResult(BaseModel):
    class_name: str = Field(..., description="The name of the detected object class.")
    confidence: float = Field(..., description="The confidence score of the detection.")
    bounding_box: List[int] = Field(..., description="The bounding box coordinates [x1, y1, x2, y2].")



class DetectionResponse(BaseModel):
    detections: List[DetectionResult] = Field(..., description="A list of detected objects.")