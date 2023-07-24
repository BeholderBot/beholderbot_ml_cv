import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import colorsys
import random
import numpy as np
from ultralytics import YOLO
import time
import math

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from vidgear.gears import CamGear

YOLO_COCO_CLASSES = "model_data/coco/coco.names"
YOLO_MODEL = "model_data/yolov8n.pt"

def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=True):   
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    return image

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def agent(tracked_bboxes_list, eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, CLASSES=YOLO_COCO_CLASSES):
    # Calculate the distance between each tensor and the eyes
    t1 = time.time()
    distances = []
    NUM_CLASS = read_class_names(CLASSES)
    for bbox in tracked_bboxes_list:
        tensor_center_x = (bbox[0] + bbox[2]) / 2
        tensor_center_y = (bbox[1] + bbox[3]) / 2
        distance_left = ((eye_x_left - tensor_center_x) ** 2 + (eye_y_left - tensor_center_y) ** 2) ** 0.5
        distance_right = ((eye_x_right - tensor_center_x) ** 2 + (eye_y_right - tensor_center_y) ** 2) ** 0.5
        distances.append((distance_left, distance_right))

    # Find the indices of the two closest tensors
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

    # Get the indices of the two closest tensors
    closest_indices = sorted_indices[:2]
    eye_left_close = False
    eye_right_close = False
    eye_label_left = ""
    eye_label_right = ""

    # Move each eye 10% of the remainder of the way closer to the respective tensors
    for i, idx in enumerate(closest_indices):
        bbox = tracked_bboxes_list[idx]
        tensor_center_x = (bbox[0] + bbox[2]) / 2
        tensor_center_y = (bbox[1] + bbox[3]) / 2 
        tensor_right_x = bbox[0]
        tensor_right_y = bbox[1]
        tensor_class = int(bbox[5])

        size = 4 * (tensor_center_x - tensor_right_x) * (tensor_center_y - tensor_right_y)

        if i == 0:
            closer_eye_x = eye_x_left
            closer_eye_y = eye_y_left
            eye_size = (left_size*2) ** 2
        else:
            closer_eye_x = eye_x_right
            closer_eye_y = eye_y_right
            eye_size = (right_size*2) ** 2

        remaining_distance = ((closer_eye_x - tensor_center_x) ** 2 + (closer_eye_y - tensor_center_y) ** 2) ** 0.5
        remaining_size = size * 0.75 - eye_size
        move_distance = remaining_distance * 0.1
        area_distance = remaining_size * 0.05

        new_eye_x = closer_eye_x - (closer_eye_x - tensor_center_x) * (move_distance / remaining_distance)
        new_eye_y = closer_eye_y - (closer_eye_y - tensor_center_y) * (move_distance / remaining_distance)
        new_eye_size = eye_size + area_distance
        eye_label = "{}".format(NUM_CLASS[tensor_class]) 

        if i == 0:
            eye_x_left = new_eye_x
            eye_y_left = new_eye_y
            left_size = int(math.sqrt(new_eye_size) / 2)
            eye_label_left = eye_label
            print(f"left eye tracking: {eye_label} at {(tensor_center_x, tensor_center_y)}")
            if move_distance <= 4:
                eye_left_close = True
        else:
            eye_x_right = new_eye_x
            eye_y_right = new_eye_y
            right_size = int(math.sqrt(new_eye_size) / 2)
            eye_label_right = eye_label
            print(f"right eye tracking: {eye_label} at {(tensor_center_x, tensor_center_y)}")
            if move_distance <= 4:
                eye_right_close = True

    t2 = time.time()

    return eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, eye_left_close, eye_right_close, eye_label_left, eye_label_right, t2 - t1

def draw_eyes(image, eye_left_close, eye_right_close, eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, eye_label_left, eye_label_right, Text_colors=(0,0,0)):
    close_dot_color = (0, 255, 0)  # Green Color
    far_dot_color = (0, 0, 255)    # Red Color
    
    # Draw the green dots representing the eyes
    if eye_left_close:
        left_dot_color = close_dot_color
    else:
        left_dot_color = far_dot_color
    if eye_right_close:
        right_dot_color = close_dot_color
    else:
        right_dot_color = far_dot_color
    
    eye_x1_left, eye_x2_left = int(eye_x_left + left_size), int(eye_x_left - left_size)
    eye_y1_left, eye_y2_left = int(eye_y_left + left_size), int(eye_y_left - left_size)
    eye_x1_right, eye_x2_right = int(eye_x_right + right_size), int(eye_x_right - right_size)
    eye_y1_right, eye_y2_right = int(eye_y_right + right_size), int(eye_y_right - right_size)

    cv2.rectangle(image, (eye_x1_left, eye_y1_left), (eye_x2_left, eye_y2_left), left_dot_color, 2)
    cv2.rectangle(image, (eye_x1_right, eye_y1_right), (eye_x2_right, eye_y2_right), right_dot_color, 2)

    # get text size
    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    fontScale = 0.75 * bbox_thick

    if eye_left_close:
        (text_width_left, text_height_left), baseline = cv2.getTextSize(eye_label_left, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=bbox_thick)
        cv2.rectangle(image, (eye_x2_left, eye_y2_left), (eye_x2_left + text_width_left, eye_y2_left - text_height_left - baseline), left_dot_color, thickness=cv2.FILLED)
        cv2.putText(image, eye_label_left, (eye_x2_left, eye_y2_left-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    if eye_right_close:
        (text_width_right, text_height_right), baseline = cv2.getTextSize(eye_label_right, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=bbox_thick)
        cv2.rectangle(image, (eye_x2_right, eye_y2_right), (eye_x2_right + text_width_right, eye_y2_right - text_height_right - baseline), right_dot_color, thickness=cv2.FILLED)
        cv2.putText(image, eye_label_right, (eye_x2_right, eye_y2_right-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    
    return image


def Object_tracking(Yolo, eye_show, track_show, CLASSES=YOLO_COCO_CLASSES):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2, times_3, times_4 = [], [], [], []

    vid = CamGear(source=0).start() 

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 

    eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, eye_label_left, eye_label_right = None, None, None, None, None, None, "", ""

    while True:
        t1 = time.time()

        image = cv2.resize(vid.read(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        pred_bbox = Yolo(image)

        t2 = time.time()
                
        tensors = [pb.boxes.data for pb in pred_bbox]
        bounding_boxes = [(x, y, w, h, confidence_score, class_id) for x, y, w, h, confidence_score, class_id in tensors[0]]
        boxes = [(x, y, w, h) for x, y, w, h, _, _ in bounding_boxes]
        scores = [confidence_score for _, _, _, _, confidence_score, _ in bounding_boxes]
        names = [class_id for _, _, _, _, _, class_id in bounding_boxes]

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(image, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        t3 = time.time()

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[int(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        # draw detection on frame
        if track_show:
            image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)

        #initialize eye areas
        if not eye_x_left:
            eye_x_left = image.shape[1] // 3
        if not eye_x_right:
            eye_x_right = 2 * image.shape[1] // 3
        if not eye_y_left:
            eye_y_left = image.shape[0] // 2
        if not eye_y_right:
            eye_y_right = image.shape[0] // 2
        if not left_size and not right_size:
            left_size = image.shape[0] // 10
            right_size = image.shape[0] // 10

        eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, eye_left_close, eye_right_close, eye_label_left, eye_label_right, agent_time = agent(tracked_bboxes, eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size)

        if eye_show:
            image = draw_eyes(image, eye_left_close, eye_right_close, eye_x_left, eye_x_right, eye_y_left, eye_y_right, left_size, right_size, eye_label_left, eye_label_right)

        cv2.imshow("Image", image)
        c = cv2.waitKey(1)
        if c == 27:
            break
        if c == ord('a'):
            eye_show = not eye_show
        if c == ord('d'):
            track_show = not track_show

        times.append(t2 - t1)
        times_2.append(t3 - t2)
        times_3.append(time.time() - t3)
        times_4.append(agent_time)

    cv2.destroyAllWindows()

    print("Average Time taken for each section (in milliseconds):")
    print("1. Resize image and YOLO inference: {:.2f} ms".format(round(sum(times) * 1000/len(times), 2)))
    print("2. deepSORT Inference: {:.2f} ms".format(round(sum(times_2) * 1000/len(times_2), 2)))
    print("3. Remainder: {:.2f} ms".format(round(sum(times_3) * 1000/len(times_3), 2)))
    print("4. Agent Hueristic: {:.2f} ms".format(round(sum(times_4) * 1000/len(times_3), 2)))

def main():
    yolo = YOLO(YOLO_MODEL)
    Object_tracking(yolo, True, False)

if __name__ == "__main__":
    main()
