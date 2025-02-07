import cv2
import easyocr

def capture_and_ocr(image_path=r"D:\pycharm\opencv\PythonProject1\testing_images\captured_image.jpg", camera_index = 0):
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None

    # Capture a frame
    ret, frame = cap.read()

    if ret:
        # Save the image
        cv2.imwrite(image_path, frame)
        print(f"✅ Image saved as {image_path}")

        # Initialize OCR reader
        reader = easyocr.Reader(['bg'])

        # Perform OCR
        result = reader.readtext(image_path)

        # Combine words into a single sentence
        sentence = " ".join([text for (bbox, text, prob) in result])

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        return sentence
    else:
        print("❌ Failed to capture image")
        cap.release()
        cv2.destroyAllWindows()
        return None
