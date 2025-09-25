
import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from utils.table_detection import TableDetector
from utils.schema import DetectionResponse
import traceback
# Initialize the FastAPI app
app = FastAPI(
    title="AI Backend Service",
    description="Performs object detection using a lightweight YOLOv8 model."
)


@app.on_event("startup")
def startup():
    global model
    model = TableDetector()


@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile = File(...)):
    """
    Receives an image, performs object detection, and returns the results.
    """
    try:

        
        # Read the image from the uploaded file
        image_bytes = await image.read()
        image_stream = io.BytesIO(image_bytes)
        
        # Open the image using PIL (Pillow)
        pil_image = Image.open(image_stream).convert("RGB")
        
        # Perform object detection inference
        detections = model.detect(np.array(pil_image))
        
        return DetectionResponse(detections=detections)

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}



if __name__ == "__main__":
    print("**************************** Starting AI backend server... **************************")
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, log_level="info")