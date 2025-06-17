from flask import Flask, render_template, request, redirect, send_from_directory, url_for, Response
import numpy as np
import cv2
import dlib
import imutils
import pytesseract
import pandas as pd
import time
import os
import math
import re

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

# Constants
WIDTH = 1280
HEIGHT = 720

# Load Haar cascade
carCascade = cv2.CascadeClassifier('myhaar.xml')

# OCR function
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
def image_to_text(image_path):
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
    cleaned_text = re.sub(r'[^A-Za-z0-9 ]+', '', raw_text).strip()

    raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'text': [cleaned_text]}
    df = pd.DataFrame(raw_data)
    df.to_csv('data.csv', mode='a', header=False, index=False)

    return cleaned_text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_speed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 12
    speed = d_meters * fps * 3.6
    return speed

def ObjectsTracking(video_path, speed_limit):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError("Unable to open video file.")

    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        start_time = time.time()
        rc, image = video.read()
        if not rc:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                        (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y = int(trackedPosition.left()), int(trackedPosition.top())
            t_w, t_h = int(trackedPosition.width()), int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        for i in carLocation1.keys():
            [x1, y1, w1, h1] = carLocation1[i]
            [x2, y2, w2, h2] = carLocation2[i]
            carLocation1[i] = [x2, y2, w2, h2]

            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                if (speed[i] is None or speed[i] == 0) and 275 <= y1 <= 285:
                    speed[i] = calculate_speed([x1, y1], [x2, y2])
                if speed[i] and speed[i] > speed_limit and y1 >= 180:
                    cv2.putText(resultImage, f"{int(speed[i])} km/hr", (int(x1 + w1 / 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    frame_filename = f"frame_{frameCounter}_car_{i}.jpg"
                    frame_path = os.path.join(OUTPUT_FOLDER, frame_filename)
                    cv2.imwrite(frame_path, resultImage)
                    image_to_text(frame_path)

    video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    vid = request.files.get('video_file')
    limit = int(request.form.get('speed_limit', 0))
    if vid and allowed_file(vid.filename):
        path = os.path.join(UPLOAD_FOLDER, vid.filename)
        vid.save(path)
        ObjectsTracking(path, limit)  # ✅ Correct usage
        return redirect(url_for('exceeded_images'))
    return "Invalid file."


@app.route('/image_process', methods=['POST'])
def image_process():
    image_file = request.files['image_file']
    if image_file and allowed_file(image_file.filename):
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
        text = image_to_text(image_path)
        return render_template('image_result.html', image_file=image_file.filename, extracted_text=text)
    return "Invalid image file."

@app.route('/exceeded_images')
def exceeded_images():
    image_filenames = [f for f in os.listdir(OUTPUT_FOLDER) if allowed_file(f)]
    return render_template('exceeded_images.html', images=image_filenames)

@app.route('/output_frames/<filename>')
def send_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)