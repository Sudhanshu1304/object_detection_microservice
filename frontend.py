import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# URL of the UI backend service
# This assumes the UI backend is running locally on port 8000
UI_BACKEND_URL = "http://localhost:8001/detect_objects/"

st.set_page_config(
    page_title="Object Detection App",
    page_icon="🤖"
)

# Set up the title and description for the app
st.title("Object Detection with YOLOv8")
st.markdown("Upload an image below to detect objects.")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner("Analyzing image..."):
        try:
            # Prepare the file for the POST request
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            # Send the image to the UI backend
            response = requests.post(UI_BACKEND_URL, files=files, timeout=60)

            if response.status_code == 200:
                # Get the detections from the response
                detections = response.json()
                if not detections:
                    st.warning("No objects were detected in the image.")
                    st.image(uploaded_file, caption="Original Image")
                else:
                    # Open the original image using Pillow
                    image_stream = io.BytesIO(uploaded_file.getvalue())
                    original_image = Image.open(image_stream).convert("RGB")
                    
                    # Create a drawing context
                    draw = ImageDraw.Draw(original_image)
                    
                    # Try to load a font for the labels, or use a default one
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()

                    # Draw the bounding boxes and labels
                    detections = detections.get('detections',[])
                    for obj in detections:
                        class_name = obj.get("class_name")
                        confidence = obj.get("confidence")
                        bounding_box = obj.get("bounding_box")
                        # Draw the bounding box rectangle
                        if class_name is not None:
                            draw.rectangle(bounding_box, outline="red", width=3)
                            
                            # Create the label text
                            label_text = f"{class_name} ({confidence:.2f})"
                            
                            # Draw a background rectangle for the label
                            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
                            draw.rectangle([bounding_box[0], bounding_box[1] - text_h - 5, bounding_box[0] + text_w + 5, bounding_box[1]], fill="red")
                            draw.text((bounding_box[0] + 2, bounding_box[1] - text_h - 2), label_text, font=font, fill="white")

                    st.success(f"Detected {len(detections)} objects!")
                    st.image(original_image, caption="Detected Objects")

            else:
                st.error(f"Error: Could not get a successful response from the backend. Status code: {response.status_code}")
                st.write(f"Response content: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: The backend service is not running. Please ensure both `ui-backend` and `ai-backend` are active.")
            st.write("You can run them using `uvicorn main:app --port 8000` and `uvicorn ai_service:app --port 8001` in separate terminals.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
