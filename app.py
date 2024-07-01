from flask import Flask, request, render_template, jsonify, redirect, url_for
import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import Counter
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Load YOLO model
model_path = r'train20/weights/best.pt'
model = YOLO(model_path)

# Initialize history storage
detection_history = []


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        results = model(image)[0]

        detections = sv.Detections.from_ultralytics(results)
        annotated_image = annotate_image(image, detections)

        chips_list, total_value = calculate_chips_and_value(detections)
        counts_text = format_counts_text(chips_list)

        output_path = os.path.join('static/uploads', 'annotated_' + filename)
        cv2.imwrite(output_path, annotated_image)

        detection_record = {
            'filename': filename,
            'counts_text': counts_text,
            'total_number': len(chips_list),
            'total_value': total_value,
            'annotated_image': output_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to history
        detection_history.append(detection_record)

        response = {
            'counts_text': counts_text,
            'total_number': f'total number of chips: {len(chips_list)}',
            'total_value': f'Total value of chips: {total_value}₪',
            'annotated_image': output_path
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/history')
def history():
    return render_template('history.html', history=detection_history)


def annotate_image(image, detections):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

    label_annotator = sv.LabelAnnotator()
    labels = [model.names[class_id] for class_id in detections.class_id]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image


def calculate_chips_and_value(detections):
    labels = [model.names[class_id] for class_id in detections.class_id]
    chips_list = []
    total_value = 0
    c1 = 1
    c25 = 0.25
    c50 = 0.5
    c500 = 2
    c5000 = 10

    for label in labels:
        chips_list.append(label)
        if label == '1':
            total_value += c1
        elif label == '25':
            total_value += c25
        elif label == '50':
            total_value += c50
        elif label == '500':
            total_value += c500
        elif label == '1000':
            total_value += c5000
    return chips_list, total_value


def format_counts_text(chips_list):
    counts = Counter(chips_list)
    counts_text = 'You have:\n'
    for element, count in counts.items():
        counts_text += f"{count}: {element}₪, \n"
    return counts_text


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))

