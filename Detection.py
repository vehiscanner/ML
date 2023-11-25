import cv2
import numpy as np

# Load YOLOv2 model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Load class names for YOLOv2 (you may need to adjust the path)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open video file
cap = cv2.VideoCapture("video.mp4")

# Initialize counters for each class of vehicle
car_count = 0
motorcycle_count = 0
bicycle_count = 0

# Create a list to store vehicle IDs that have crossed the counting line
crossed_line_ids = []

# Define the counting line (centered in the frame)
counting_line = [(35, 600), (1250, 600)]  # Adjust the coordinates as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Draw a line in the center of the frame
    line_color = (0, 0, 255)  # Red color
    line_thickness = 2
    center_line_y = counting_line[0][1]
    cv2.line(frame, counting_line[0], counting_line[1], line_color, line_thickness)

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass
    outs = net.forward(layer_names)

    # Initialize lists to store detected objects and their dimensions
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            if class_name in ['car', 'motorbike', 'bicycle']:
                # Calculate the length and width of the detected object
                length = w
                width = h

                # Check if the vehicle has crossed the counting line
                if (x, y) not in crossed_line_ids and y + h > center_line_y and y < center_line_y:
                    crossed_line_ids.append((x, y))
                    
                    # Count the number of each class of vehicle
                    if class_name == 'car':
                        car_count += 1
                    elif class_name == 'motorbike':
                        motorcycle_count += 1
                    elif class_name == 'bicycle':
                        bicycle_count += 1

                # Draw a bounding box and label on the frame
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{class_name} ({confidences[i]:.2f}), Length: {length}, Width: {width}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the count on the video
    cv2.putText(frame, f"Total Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Motorbike: {motorcycle_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Bicycles: {bicycle_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Video with Detections", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
