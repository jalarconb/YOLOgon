import cv2
import time
import os


def capture_images(output_folder="raw_data", num_images=200, delay=0.5):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(0)  # Start webcam
    if not cap.isOpened():
        print("ERROR: Unable to start the camera.")
        return

    print(f"Capturing {num_images} images. Press ENTER to start...")
    input()

    for i in range(1, num_images + 1):
        ret, frame = cap.read()
        if not ret:
            print("ERROR when trying to capture the image")
            break

        image_path = os.path.join(output_folder, f"img_{i}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image {i}/{num_images} saved in {image_path}")

        time.sleep(delay)  # Wait between captures

    cap.release()
    cv2.destroyAllWindows()
    print("Capture complete.")


# --------------------------------------------------------------------------------

# Image Capture

capture_images(output_folder="raw_data/images/no_glasses")
capture_images(output_folder="raw_data/images/with_glasses")
