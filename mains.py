import cv2
import numpy as np
from fpdf import FPDF

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


def create_indian_flag_with_face(face_img):
    # Flag dimensions
    width, height = 700, 300  # 3:2 ratio

    # Create blank flag image
    flag = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw saffron band (top)
    flag[0:height // 3, :] = (0, 140, 255)  # BGR for saffron

    # Draw white band (middle)
    flag[height // 3:2 * height // 3, :] = (255, 255, 255)

    # Draw green band (bottom)
    flag[2 * height // 3:, :] = (0, 128, 0)

    # Draw Ashoka Chakra (navy blue circle) in the center of white band
    center = (width // 2, height // 2)
    radius = height // 6
    navy_blue = (128, 0, 0)  # BGR approximation for navy blue

    cv2.circle(flag, center, radius, navy_blue, thickness=8)

    # Draw 24 spokes of the chakra
    for i in range(24):
        angle = i * 15  # degrees
        x2 = int(center[0] + radius * np.cos(np.radians(angle)))
        y2 = int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.line(flag, center, (x2, y2), navy_blue, 2)

    # Resize face image to fit inside chakra circle
    face_w = radius * 2
    face_h = radius * 2
    face_resized = cv2.resize(face_img, (face_w, face_h))

    # Create a circular mask to crop face into circle shape
    mask = np.zeros((face_h, face_w), dtype=np.uint8)
    cv2.circle(mask, (face_w // 2, face_h // 2), radius, 255, -1)

    # Prepare region of interest (ROI) on flag
    y1 = center[1] - radius
    y2 = center[1] + radius
    x1 = center[0] - radius
    x2 = center[0] + radius

    roi = flag[y1:y2, x1:x2]

    # Mask face and background
    face_fg = cv2.bitwise_and(face_resized, face_resized, mask=mask)
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    # Add face on chakra circle area
    dst = cv2.add(roi_bg, face_fg)
    flag[y1:y2, x1:x2] = dst

    return flag


def save_image_as_pdf(image, pdf_path):
    temp_img_path = "temp_capture.png"
    cv2.imwrite(temp_img_path, image)

    from fpdf import FPDF
    pdf = FPDF(unit="pt", format=[image.shape[1], image.shape[0]])
    pdf.add_page()
    pdf.image(temp_img_path, 0, 0, image.shape[1], image.shape[0])
    pdf.output(pdf_path)


face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y + h, x:x + w]

        # Create flag graphic with face embedded
        final_flag = create_indian_flag_with_face(face_img)

        # Save PDF automatically and exit
        save_image_as_pdf(final_flag, "face_in_flag_graphic.pdf")
        print("Face detected and saved as flag_graphic.pdf")

        break

cap.release()
cv2.destroyAllWindows()
