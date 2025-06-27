import cv2
import torch
import numpy as np
from PIL import Image
from gtts import gTTS
import os
from ultralytics import YOLO

# Check if CUDA (GPU) is available
device = "cpu"  # Use CPU since GPU is not available
print(f"Using device: {device}")


# Load YOLOv5 Model (Pre-trained on COCO Dataset)
model = YOLO("yolov5s.pt").to(device)
 # Move model to GPU if available

# Object descriptions
object_descriptions = {
    "person": "A human being detected.",
    "car": "A vehicle used for transportation.",
    "dog": "A domestic animal known for loyalty.",
    "cat": "A small domesticated carnivorous mammal.",
    "bottle": "A container used to hold liquids.",
    "chair": "A piece of furniture used for sitting.",
    "cup": "A small open container used for drinking.",
    "cell phone": "A handheld device for communication.",
}

# Function to get dominant color
def get_dominant_color(image):
    """Returns the dominant color in an image as an RGB tuple."""
    image = image.resize((50, 50))  # Resize for efficiency
    pixels = np.array(image).reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_color = unique_colors[counts.argmax()]  # Most frequent color
    return tuple(dominant_color)

# Function to map RGB to color name
def rgb_to_color_name(rgb):
    """Maps RGB values to the closest known color."""
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
    }
    min_distance = float("inf")
    color_name = "unknown"
    for name, value in colors.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(value))
        if distance < min_distance:
            min_distance = distance
            color_name = name
    return color_name

# Function for voice feedback
import tempfile
import os
from gtts import gTTS
from playsound import playsound  # Import playsound for proper audio playback

def text_to_speech(text):
    """Generates and plays voice feedback safely."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name  # Generate temporary path
        tts = gTTS(text=text, lang="en")
        tts.save(temp_audio_path)  # Save file

    playsound(temp_audio_path)  # Ensures the audio plays fully before proceeding

    os.remove(temp_audio_path)  # Cleanup after playback



# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB and detect objects
    results = model(frame, imgsz=320, half=True)  # Reduce image size & use half precision


    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract object image and detect color
            object_image = frame[ymin:ymax, xmin:xmax]
            if object_image.size != 0:
                object_image_pil = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
                dominant_color = get_dominant_color(object_image_pil)
                color_name = rgb_to_color_name(dominant_color)

                # Display color information
                color_label = f"Color: {color_name}"
                cv2.putText(frame, color_label, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Get description
                description = object_descriptions.get(class_name, "No description available.")

                # Voice feedback
                feedback = f"Detected {class_name} with {confidence * 100:.2f}% confidence. The color is {color_name}. {description}"
                text_to_speech(feedback)

    # Show the video frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
