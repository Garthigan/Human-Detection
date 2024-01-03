import cv2
import numpy as np

# Load YOLOv4 model with coco dataset for people detection
net = cv2.dnn.readNetFromDarknet("/Users/garthigan.k/Desktop/CV/Human-Detection/yolov4.cfg", "/Users/garthigan.k/Desktop/CV/Human-Detection/yolov4.weights")
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Convert indices to layer names
# Convert indices to layer names
output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]


# Initialize labels list
labels = ['person']

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

def detect_people(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and labels[class_id] == "person":
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                
    return boxes

def resize_frame(frame, scale_percent=50):
    """
    Resize the frame to reduce its quality.
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

video_path = '/Users/garthigan.k/Desktop/CV/Human-Detection/sample.mp4'
cap = cv2.VideoCapture(video_path)

tracker = cv2.TrackerCSRT_create()
bbox = None
object_id = 0
object_dict = {}
smooth_count = 0

frame_skip = 5
current_frame = 0

while True:
    ret, frame = cap.read()
    current_frame += 1
    
    if current_frame % frame_skip != 0:
        continue
    
    if not ret:
        cap.release()
        cap = cv2.VideoCapture(video_path)
        continue
    
    resized_frame = resize_frame(frame)
    boxes = detect_people(resized_frame)
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.putText(resized_frame, f'People Count: {len(boxes)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Resized Frame', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
