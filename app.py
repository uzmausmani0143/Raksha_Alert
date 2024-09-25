from flask import Flask, Response, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename  # Add this import
from flask import send_from_directory
import cv2
import face_recognition
import os
from datetime import datetime
import geocoder
from deepface import DeepFace
import numpy as np
import pygame
import threading
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import speech_recognition as sr
import boto3

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'your_generated_secret_key'  # Update this with a generated key

sns_client= boto3.client('sns', region_name='us-east-1')

#ARNs
arn_police = 'arn:aws:sns:us-east-1:107335645910:sumanbot'
arn_neighbors = 'arn:aws:sns:us-east-1:107335645910:Padosi'

#Intialiaze recognizer
recognizer=sr.Recognizer()

# Initialize pygame mixer for sound control
pygame.mixer.init()

# Initialize face recognition
known_face_encodings = []
known_face_names = []
captured_unknown_face_encodings = []

# Directory containing images of known persons
known_faces_dir = r"C:\Users\Acer\Downloads\All_Work\Raksha_Alert\known"

# Directory to save images of unknown persons
unknown_faces_dir = r"C:\Users\Acer\Downloads\All_Work\Raksha_Alert\unknown"

# Email configuration
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_user = "uzmausmani0143@gmail.com"
smtp_password = "rfba mkle zxwe krrw"
recipient_email = "priyankabharti0818@gmail.com"

# Load each image file and extract face encodings
for image_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, image_name)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(image_name)[0])

# # Global variable to store geolocation information
location_info = None

# def fetch_geolocation():
#     global location_info
#     # Fetch geolocation
#     g = geocoder.ip('me')
#     location_info = f"Location: {g.city}, {g.state}, {g.country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"

# # Call the function to fetch geolocation info once
# fetch_geolocation()


def fetch_geolocation():
    global location_info
    try:
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            city = g.city if g.city else "Unknown City"
            state = g.state if g.state else "Unknown State"
            country = g.country if g.country else "Unknown Country"
            location_info = f"Location: {city}, {state}, {country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"
        else:
            location_info = "Geolocation data not available"
    except Exception as e:
        location_info = f"Error fetching geolocation: {str(e)}"

fetch_geolocation()

# Load YOLO object detection model
model_config = "yolov4.cfg"
model_weights = "yolov4.weights"
coco_names = "coco.names"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
layer_names = net.getLayerNames()

# Handle different versions of OpenCV
out_layer_indices = net.getUnconnectedOutLayers()
if len(out_layer_indices.shape) == 1:
    output_layers = [layer_names[i - 1] for i in out_layer_indices]
else:
    output_layers = [layer_names[i[0] - 1] for i in out_layer_indices]

with open(coco_names, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

dangerous_objects = ["knife", "gun", "scissors"]
alarm_playing = False

def play_alarm():
    global alarm_playing
    sound_file = 'alert.mp3'
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(-1)
    alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def start_alarm():
    global alarm_playing
    if not alarm_playing:
        play_alarm()
        threading.Timer(10.0, stop_alarm).start()

def send_email(unknown_image_path):
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient_email
        msg['Subject'] = 'Unknown Face Detected'

        body = f'An unknown face has been detected and saved as an image.\n\n{location_info}'
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image
        with open(unknown_image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(unknown_image_path)}')
            msg.attach(part)

        # Send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def generate_frames():
    global alarm_playing
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        try:
            faces = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
                if not any(face_recognition.compare_faces(captured_unknown_face_encodings, face_encoding, tolerance=0.6)):
                    top, right, bottom, left = face_location
                    face_image = frame[top:bottom, left:right]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_image_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(unknown_image_path, face_image)
                    captured_unknown_face_encodings.append(face_encoding)

                    # Send email with unknown face image
                    send_email(unknown_image_path)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, width, height = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if classes[class_ids[i]] in dangerous_objects:
                    if not alarm_playing:
                        start_alarm()

        for face in faces:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, face['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (54, 219, 9), 2)


        (text_width, text_height), baseline = cv2.getTextSize(location_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Draw a rectangle behind the text
        cv2.rectangle(frame, (10, frame.shape[0] - 10 - text_height - baseline), (10 + text_width, frame.shape[0] - 10 + baseline), (100,100,102), cv2.FILLED) #grey 
        cv2.putText(frame, location_info, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2) #green

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

def send_sns_alert(arn, message):
    try:
        sns_client.publish(
            TopicArn=arn,
            Message=message,
            Subject='Emergency Alert'
        )
        print(f"Alert sent to ARN: {arn}")
    except Exception as e:
        print(f"Error sending SNS alert: {e}")

def listen_for_commands():
    with sr.Microphone() as source:
        print("Listening for voice commands...")
        while True:
            try:
                # Adjust for ambient noise and listen
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

                # Recognize speech
                command = recognizer.recognize_google(audio).lower()

                # Check for specific keywords to trigger SNS alerts
                if 'laptop' in command:
                    send_sns_alert(arn_police, 'Alert: Assistance required from the police.')
                elif 'picnic' in command or 'neighbor' in command:
                    send_sns_alert(arn_neighbors, 'Alert: Assistance required from neighbors.')
                else:
                    print(f"Unrecognized command: {command}")

            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")

@app.route('/known/<filename>')
def serve_known_image(filename):
    return send_from_directory(known_faces_dir, filename)

@app.route('/add_member', methods=['POST'])
def add_member():
    if 'member_image' not in request.files or 'member_name' not in request.form:
        flash('No file or member name provided.', 'error')
        return redirect(url_for('member'))

    file = request.files['member_image']
    member_name = request.form['member_name']

    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('member'))

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(known_faces_dir, filename)
        file.save(file_path)

        image = face_recognition.load_image_file(file_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(member_name)
            flash('Member added successfully.', 'success')
        else:
            flash('No face found in the image.', 'error')
            os.remove(file_path)

    return redirect(url_for('member'))

# Delete Member
@app.route('/delete_member/<filename>', methods=['POST'])
def delete_member(filename):
    try:
        # Construct the full path to the file
        file_path = os.path.join(known_faces_dir, filename)
        
        # Delete the file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            # Also remove face encoding from the list
            global known_face_encodings, known_face_names
            try:
                index = known_face_names.index(os.path.splitext(filename)[0])
                del known_face_encodings[index]
                del known_face_names[index]
            except ValueError:
                pass  # The name wasn't in the list
            flash('Member image deleted successfully.', 'success')
        else:
            flash('Image not found.', 'error')
    except Exception as e:
        flash(f'Error deleting image: {e}', 'error')

    return redirect(url_for('member'))

@app.route('/unknown')
def unknown():
    # List all files in the unknown folder
    unknown_images_list = [f for f in os.listdir(unknown_faces_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return render_template('unknown.html', unknown=unknown_images_list)

@app.route('/delete_unknown_image/<filename>', methods=['POST'])
def delete_unknown_image(filename):
    try:
        # Construct the full path to the file
        file_path = os.path.join(unknown_faces_dir, filename)
        
        # Delete the file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            flash('Image deleted successfully.', 'success')
        else:
            flash('Image not found.', 'error')
    except Exception as e:
        flash(f'Error deleting image: {e}', 'error')

    return redirect(url_for('unknown'))

@app.route('/unknown/<filename>')
def serve_unknown_image(filename):
    return send_from_directory(unknown_faces_dir, filename)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/member')
def member():
    known_images = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return render_template('member.html', known_images=known_images)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = known_faces_dir
    app.run(debug=True)
