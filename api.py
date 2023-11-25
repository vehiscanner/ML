from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Menerima gambar dari klien
        image = request.files['image'].read()
        nparr = np.fromstring(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        height, width = frame.shape[:2]

        # Create a blob from the frame
        blob = cv2.dnn.blobFromI
        image(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

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

        results = []

        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                class_name = classes[class_ids[i]]
                if class_name in ['car', 'motorbike', 'bicycle']:
                    # Calculate the length and width of the detected object
                    length = w
                    width = h

                    # Count the number of each class of vehicle
                    if class_name == 'car':
                        results.append({"Class": "car", "Length": length, "Width": width})
                    elif class_name == 'motorbike':
                        results.append({"Class": "motorbike", "Length": length, "Width": width})
                    elif class_name == 'bicycle':
                        results.append({"Class": "bicycle", "Length": length, "Width": width})

        return jsonify({"objects": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
