from flask import Flask, render_template, request, jsonify, Response 
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the face recognition model
MODEL_PATH = "efficientnet_face_recognition_with_mobilenet.keras"
model = tf.keras.models.load_model(MODEL_PATH)
data = 'attendance.csv'
# Define dataset directory
DATASET_PATH = "dataset"
MAX_IMAGES = 400

# Get class names from dataset
class_names = sorted(os.listdir(DATASET_PATH))
num_model_classes = model.output_shape[-1]
if len(class_names) != num_model_classes:
    print(f"Warning: Model classes ({num_model_classes}) don't match dataset classes ({len(class_names)})")
class_names.append("Unidentified")

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

# --------------- VIDEO STREAMING ----------------
def generate_frames():
    """ Generates frames for face capture. """
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """ Video feed route for face capture. """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_attendance_frames():
    """ Generates frames for attendance recognition. """
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0  # Normalize

            face = np.expand_dims(face, axis=0)
            predictions = model.predict(face)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)

            threshold = 0.7
            is_identified = confidence >= threshold
            if 0 <= class_id < len(class_names):  # Ensure index is within range
                class_label = class_names[class_id] if is_identified else "Unidentified"
            else:
                class_label = "Unidentified"

            if class_label == "Unidentified":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed_attendence')
def video_feed_attendance():
    """ Video feed route for attendance recognition. """
    return Response(generate_attendance_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --------------- FACE CAPTURE ----------------
@app.route('/capture', methods=['POST'])
def capture():
    """ Captures face images and saves them to dataset. """
    person_name = request.json.get("name")
    if not person_name:
        return jsonify({"error": "Missing name"}), 400

    save_path = os.path.join(DATASET_PATH, person_name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (224, 224))

            filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(filename, face_resized)
            count += 1
            if count >= MAX_IMAGES:
                break
        
        time.sleep(0.2)  

    return jsonify({"message": f"Captured {count} images for {person_name}"})

# --------------- ATTENDANCE RECOGNITION ----------------


@app.route('/capture_attendance', methods=['POST'])
def capture_attendance():
    """ Captures attendance by recognizing a face. """
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"message": "No face detected"}), 200

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0  # Normalize

        face = np.expand_dims(face, axis=0)
        predictions = model.predict(face)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)

        threshold = 0.7
        is_identified = confidence >= threshold
        if 0 <= class_id < len(class_names):  # Ensure index is within range
            class_label = class_names[class_id] if is_identified else "Unidentified"
        else:
            class_label = "Unidentified"

        if class_label != "Unidentified":
            new_entry = {"Name": class_label, "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            df_new = pd.DataFrame([new_entry])
            df_new.to_csv(data, mode='a', header=False, index=False)

            df = pd.read_csv(data)
            df['Date'] = pd.to_datetime(df['Date']) 

            df['Hour'] = df['Date'].dt.strftime('%Y-%m-%d %H')  
            df = df.sort_values('Date').drop_duplicates(subset=['Name', 'Hour'], keep='first')

            df = df.drop(columns=['Hour'])

            df.to_csv(data, index=False)
            return jsonify({"message": f"Attendance recorded for {class_label}"}), 200
        else:
            return jsonify({"message": f"Unidentified Person"}), 200

    return jsonify({"message": "Face detected but not recognized"}), 200

# --------------- TENSORFLOW WORK ----------------

def train_model():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
)

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )

    class_names = train_dataset.class_names
    print("Class names:", class_names)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2)
    ])

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y)
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False 

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ]) 

    model.compile(optimizer=optimizers.Adam(learning_rate=0.00005), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])


    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[early_stopping])


    model.save("efficientnet_face_recognition_with_mobilenet.keras")



@app.route('/train_model', methods=['POST'])
def train():
    return Response(train_model(), content_type='text/html')

@app.route("/table")
def table():
    df = pd.read_csv('attendance.csv') 
    table = df.to_html(classes='table table-striped', index=False)  
    return render_template('viewlist.html', table=table) 
    
# --------------- HTML ROUTES ----------------
@app.route('/')
def index():
    return render_template('htmlface.html')

@app.route('/attendance')
def attendance():
    return render_template('attendence.html')

@app.route('/capture')
def capture_face():
    return render_template('app.html')

# --------------- START SERVER ----------------
if __name__ == '__main__':
    app.run(debug=True)
