from flask import Flask, request, render_template, jsonify
import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load YOLO model
model_path = r'train20/weights/best.pt'
model = YOLO(model_path)


@app.route('/')
def index():
    return render_template('index.html')


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

        response = {
            'counts_text': counts_text,
            'total_number': f'total number of chips: {len(chips_list)}',
            'total_value': f'Total value of chips: {total_value}₪',
            'annotated_image': output_path
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


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
    for label in labels:
        chips_list.append(label)
        if label == '1':
            total_value += int(label)
        else:
            total_value += (int(label) / 100)
    return chips_list, total_value


def format_counts_text(chips_list):
    counts = Counter(chips_list)
    counts_text = 'You have:\n'
    for element, count in counts.items():
        pluralized_element = f"{element}s" if count > 1 else element
        counts_text += f"{count}: {pluralized_element}\n"
    return counts_text


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
