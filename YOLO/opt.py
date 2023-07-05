import cv2
import heuristic
from ultralytics import YOLO
from vidgear.gears import CamGear

cap = CamGear(source=0).start() 

# define some constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)

model = YOLO("yolov8s.pt")
render = 0

# Specify the number of cells vertically and horizontally
h = 10  # Number of cells vertically
w = 10  # Number of cells horizontally

while True:
    frame = cap.read()
    if frame is None:
        break
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    if render % 4 == 0:
        detections = model(frame)[0]
    render += 1

    # Retrieve bounding box coordinates and class labels
    boxes = detections.boxes  # Bounding box coordinates
    labels = detections.names

    # Calculate cell width and height
    cell_height = frame.shape[0] // h
    cell_width = frame.shape[1] // w

    # Draw cell boundaries
    for i in range(h):
        for j in range(w):
            cv2.rectangle(frame, (j * cell_width, i * cell_height),
                          ((j + 1) * cell_width, (i + 1) * cell_height), GREEN, 1)

    # Iterate over each detected object
    for box in boxes.data:
        x1, y1, x2, y2, confidence, label = box.tolist()
        if confidence > CONFIDENCE_THRESHOLD:
            # Calculate the cell coordinates for the bounding box
            cell_x = int(x1) // cell_width
            cell_y = int(y1) // cell_height

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
            cv2.putText(frame, labels[label], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

    cv2.imshow("Image", frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.stop()
cv2.destroyAllWindows()


#TODO  What I want is to add a simple heuristic that returns the zoomed in subpatches of the processed image whereever there is a detected object

