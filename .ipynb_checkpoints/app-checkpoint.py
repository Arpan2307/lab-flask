from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import easyocr
import re
import matplotlib.pyplot as plt
import pickle
from werkzeug.utils import secure_filename

os.environ['NO_PROXY'] = '127.0.0.1'




# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained models
with open('model1.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('model2.pkl', 'rb') as f:
    model2 = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            fields, processed_image = process_screenshot(filepath)
            predictions = predict(fields)
            return render_template('output.html', fields=fields, predictions=predictions, image_path=filepath)
    return render_template('upload.html')

def process_screenshot(image_path):
    fields = {}

    # Your processing logic
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Extract text
    extracted_text = extract_text_easyocr(image_path)
    result = extract_digit_with_following_chars(extracted_text)
    if len(result) == 0:
        result.append(('1', 'km'))

    y_scale = int(0.97 * height)
    x_scale = int(0.83 * width)
    roi = image[y_scale:height, x_scale:width]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
    edges_roi = cv2.Canny(binary_roi, 50, 150)
    contours_roi, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_length = 0
    scale_bar_contour = None

    for contour in contours_roi:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > max_length and h > 5 and area > 20:  
            max_length = w
            scale_bar_contour = contour

    # Find icons
    small_image_path = "static/img/red_icon.png"
    red_icon_coords = find_image_coordinates(image_path, small_image_path)
    small_image_path = "static/img/white_icon.png"
    white_icon_coords = find_image_coordinates(image_path, small_image_path)

    # Calculate aerial distance
    ad = aerial_distance(white_icon_coords, red_icon_coords)

    # Color detection
    color_image_path = "static/img/red_icon.png"
    red_mask, screenshot_rgb, rp = find_color_pixels(image_path, color_image_path)
    color_image_path = "static/img/yellow_icon.png"
    yellow_mask, screenshot_rgb, yp = find_color_pixels(image_path, color_image_path)
    color_image_path = "static/img/blue_icon.png"
    blue_mask, screenshot_rgb, bp = find_color_pixels(image_path, color_image_path)

    # Scale conversion
    result[-1] = combine_tuple_elements(result[-1])
    scale_bar_length_m = scale_in_meters(result[-1])
    meters_per_pixel = scale_bar_length_m / max_length
    ad_m = ad * meters_per_pixel

    # Populate fields
    fields['max_length'] = max_length
    fields['scale_bar_length_m'] = scale_bar_length_m
    fields['meters_per_pixel'] = meters_per_pixel
    fields['ad'] = ad
    fields['ad_m'] = ad_m
    fields['rp'] = rp
    fields['yp'] = yp
    fields['bp'] = bp
    fields['red_icon_coords'] = red_icon_coords
    fields['white_icon_coords'] = white_icon_coords

    return fields, image

def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)

def extract_digit_with_following_chars(text):
    pattern = r'\s?\b(1|2|5|7|10|20|50|100|200|500)\s?(km|m)\b'
    matches = re.findall(pattern, text)
    return matches

def combine_tuple_elements(tup):
    return ' '.join(tup)

def scale_in_meters(text):
    dis = {
        "1 km": 1000, "1km": 1000, "2 km": 2000, "2km": 2000,
        "5 km": 5000, "5km": 5000, "7 km": 7000, "7km": 7000,
        "10 m": 10, "10m": 10, "20 m": 20, "20m": 20,
        "50 m": 50, "50m": 50, "100 m": 100, "100m": 100,
        "200 m": 200, "200m": 200, "500 m": 500, "500m": 500
    }
    return dis.get(text, 0)

def find_image_coordinates(large_image_path, small_image_path):
    large_image = cv2.imread(large_image_path)
    small_image = cv2.imread(small_image_path)

    large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(large_gray, small_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + small_image.shape[1], top_left[1] + small_image.shape[0])
    return top_left, bottom_right

def aerial_distance(white_icon_coords, red_icon_coords):
    if white_icon_coords is None or red_icon_coords is None:
        return 0
    x1, y1 = white_icon_coords[0]
    x2, y2 = red_icon_coords[0]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def find_color_pixels(screenshot_path, color_image_path):
    screenshot = cv2.imread(screenshot_path)
    color_image = cv2.imread(color_image_path)

    if screenshot is None or color_image is None:
        print("Error: Could not load images.")
        return None, None, 0

    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    average_color = np.mean(color_image_rgb, axis=(0, 1))

    color_range = 5
    lower_bound = np.maximum(average_color - color_range, 0)
    upper_bound = np.minimum(average_color + color_range, 255)

    mask = cv2.inRange(screenshot_rgb, lower_bound, upper_bound)
    matching_pixels = cv2.countNonZero(mask)
    return mask, screenshot_rgb, matching_pixels

def predict(fields):
    field_values = np.array(list(fields.values())).reshape(1, -1)
    
    # Example: Use model1 to predict original distance
    original_distance = model1.predict(field_values)[0]
    
    # Example: Use model2 to predict estimated time
    estimated_time = model2.predict(field_values)[0]
    
    return {
        'original_distance': original_distance,
        'estimated_time': estimated_time
    }

if __name__ == '__main__':
    app.run(debug=False)
