import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# define some constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)

model = YOLO("yolov8s.pt")
render = 0
while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if render%4 == 0:
        detections = model(frame)[0]
    render += 1

    # Retrieve bounding box coordinates and class labels
    boxes = detections.boxes  # Bounding box coordinates
    labels = detections.names

    # Iterate over each detected object
    for box in boxes.data:
        x1, y1, x2, y2, confidence, label = box.tolist()
        if confidence > CONFIDENCE_THRESHOLD:
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
            cv2.putText(frame, labels[label], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

    cv2.imshow("Image", frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()