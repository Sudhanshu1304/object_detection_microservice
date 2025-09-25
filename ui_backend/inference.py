import os
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Initialize the FastAPI app
app = FastAPI(
    title="UI Backend Service",
    description="Handles image uploads and forwards them to the AI backend for object detection."
)

# The URL of the AI backend service.
# In a Docker Compose network, services can communicate using their service names.
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://localhost:8000/detect/")
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"]

@app.post("/detect_objects")
async def detect_objects(image: UploadFile = File(...)):
    """
    Receives an image, sends it to the AI backend for object detection,
    and returns the structured results.
    """
    try:
        if image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {image.content_type}. Please upload a JPEG or PNG image."
            )
        # Read the image content
        image_bytes = await image.read()

        # Prepare the file for the POST request to the AI backend
        files = {"image": (image.filename, image_bytes, image.content_type)}
        # Forward the image to the AI backend
        response = requests.post(AI_BACKEND_URL, files=files, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        # Return the JSON response from the AI backend
        return JSONResponse(content=response.json(), status_code=response.status_code)

    except requests.exceptions.RequestException as e:
        # Handle connection errors or other request-related issues
        return JSONResponse(
            content={"error": f"Failed to connect to the AI backend: {e}"},
            status_code=503
        )
    except Exception as e:
        # Handle any other unexpected errors
        return JSONResponse(
            content={"error": f"An unexpected error occurred: {e}"},
            status_code=500
        )



if __name__ == "__main__":
    print("**************************** Starting UI backend server... **************************")
    uvicorn.run("inference:app", host="0.0.0.0", port=8001, log_level="info")