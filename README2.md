

# Traffic information Extractor app

## Overview

This project is a Flask-based web application that processes uploaded images to extract specific features and make predictions based on those features. The application can detect specific colors, calculate distances between icons, and predict certain metrics using pre-trained machine learning models.

## Features

- Upload image files in various formats (PNG, JPG, JPEG, GIF).
- Process images to extract features such as:
  - Maximum length of a scale bar in the image.
  - Meters per pixel.
  - Aerial distance and aerial distance in meters.
  - Count of red, yellow, and blue pixels.
- Make predictions for:
  - Original Distance
  - Estimated Time
- Display the processed image and results in a user-friendly interface.

## Setup and Installation

## Prerequisites

- Python 3.x
- Flask
- OpenCV
- EasyOCR
- NumPy
- Werkzeug
- Pickle (for loading pre-trained models)
- Ensure `static/uploads` and `static/img` directories exist

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-processing-prediction-app.git
   cd image-processing-prediction-app
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your pre-trained models (`scaler1.pkl`, `model1.pkl`, `model2.pkl`, and `scaler.pkl`) in the root directory.

4. Ensure you have the icon and color reference images in `static/img`:
   - `red_icon.png`
   - `white_icon.png`
   - `red.png`
   - `yellow.png`
   - `blue.png`

## Running the Application

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

## Usage

1. Navigate to the home page and click on "Upload Image".
2. Select an image file and upload it.
3. The application will process the image and display the results including the processed image, extracted fields, and predictions.

## Project Structure

- `app.py`: Main Flask application file.
- `templates/`: HTML templates for the application.
  - `index.html`: Home page template.
  - `upload.html`: File upload template.
  - `output.html`: Result display template.
- `static/uploads/`: Directory to store uploaded and processed images.
- `static/img/`: Directory for reference images used in processing.

## Important Functions

- `process_screenshot(image_path)`: Processes the uploaded image to extract features and return them.
- `extract_text_easyocr(image_path)`: Uses EasyOCR to extract text from the image.
- `extract_digit_with_following_chars(text)`: Extracts digits followed by specific units from text.
- `combine_tuple_elements(tup)`: Combines elements of a tuple into a single string.
- `scale_in_meters(text)`: Converts scale text to meters.
- `find_image_coordinates(large_image_path, small_image_path)`: Finds coordinates of a small image within a large image.
- `aerial_distance(white_icon_coords, red_icon_coords)`: Calculates the distance between two points.
- `find_color_pixels(screenshot_path, color_image_path)`: Finds the number of pixels matching a color reference image.
- `predict(fields)`: Makes predictions using pre-trained models.

## Troubleshooting

- Ensure all required packages are installed.
- Check the paths for pre-trained models and reference images.
- Verify the Flask server is running and accessible at the specified URL.

## Contributing

Feel free to fork this repository and submit pull requests. Any improvements or bug fixes are welcome!

