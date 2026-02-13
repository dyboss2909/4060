import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set figure size for display
plt.rcParams['figure.figsize'] = (18.0, 10.0)

# --- Step 1: Read Image ---
# Replace "QRCode.png" with your actual file path if different
img_path = "QRCode.png"
img = cv2.imread(img_path)

if img is None:
    print("Error: Could not read image.")
else:
    # Convert to RGB for correct matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Step 2: Detect QR Code in the Image ---
    # Instantiate the QRCodeDetector object
    qr_decoder = cv2.QRCodeDetector()

    # Detect and decode
    # opencvData: The decoded string
    # bbox: The array of vertices of the detected QR code
    # rectifiedImage: The processed/aligned QR code image
    opencvData, bbox, rectifiedImage = qr_decoder.detectAndDecode(img)

    if opencvData:
        print(f"QR Code Detected!")
        print(f"Decoded Data: {opencvData}")

        # --- Step 3: Draw bounding box around the detected QR Code ---
        # bbox returns corners in (x, y) format. 
        # According to your lecture notes on Pixel Relations, 
        # we connect these coordinates to define the boundary.
        if bbox is not None:
            n = len(bbox[0])
            for i in range(n):
                # Draw lines between the corners
                # We use bbox[0][i] and the next point (modulo n to wrap back to start)
                pt1 = tuple(bbox[0][i].astype(int))
                pt2 = tuple(bbox[0][(i + 1) % n].astype(int))
                cv2.line(img_rgb, pt1, pt2, color=(0, 255, 0), thickness=3)

        # --- Step 4: Print the Decoded Text ---
        # (This is handled by the print statements above)

        # --- Step 5: Save and display the result image ---
        # Convert back to BGR to save correctly with OpenCV
        result_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("QRCode_Detected_Result.png", result_bgr)

        # Display the result
        plt.imshow(img_rgb)
        plt.title("Detected QR Code")
        plt.axis('off')
        plt.show()

    else:
        print("QR Code NOT Detected")