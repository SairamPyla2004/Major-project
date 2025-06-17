from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
import os
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to perform image processing and text extraction
def image_to_text(image_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = cv2.imread(image_path)

    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 170, 200)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    config = ('-l eng --oem 1 --psm 3')
    raw_text = pytesseract.image_to_string(new_image, config=config)

    # Clean the extracted text
    cleaned_text = re.sub(r'[^A-Za-z0-9 ]+', '', raw_text).strip()

    raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'text': [cleaned_text]}
    df = pd.DataFrame(raw_data)
    df.to_csv('data.csv', mode='a', header=False, index=False)

    return new_image, cleaned_text

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return render_template('index.html', error="No image part")

        file = request.files['image_file']

        if file.filename == '':
            return render_template('index.html', error="No selected image")

        # Ensure the uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(image_path)

        # Perform image processing and text extraction
        processed_image, extracted_text = image_to_text(image_path)

        # Save the processed image
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
        cv2.imwrite(processed_image_path, processed_image)

        # Render the results in image_result.html
        return render_template('image_result.html', uploaded_image='uploaded_image.jpg', processed_image='processed_image.jpg', extracted_text=extracted_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)