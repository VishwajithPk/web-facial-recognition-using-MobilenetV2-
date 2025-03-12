# Face Recognition-Based Attendance System

## Overview
This project is an AI-powered **Face Recognition-Based Attendance System** that automates student attendance tracking using **deep learning** and **computer vision**. It leverages **Flask, OpenCV, TensorFlow**, and a **MobileNetV2** model for real-time face recognition.

## Features
- **Automated Attendance**: Recognizes students and logs their attendance.
- **Web Interface**: Built using HTML, JavaScript, and Flask.
- **Real-Time Face Recognition**: Uses OpenCV for live detection.
- **Model Training**: Implements MobileNetV2 for deep learning-based face recognition.
- **Attendance Storage**: Saves logs in a CSV file for easy retrieval.
- **API Support**: RESTful API endpoints for face capture, recognition, and attendance tracking.

## Tech Stack
- **Frontend**: HTML, JavaScript
- **Backend**: Flask, Python
- **Machine Learning**: TensorFlow, OpenCV
- **Database**: CSV for attendance storage

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- OpenCV
- TensorFlow
- NumPy
- Pandas

### Steps to Run the Project
1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-username/face-recognition-attendance.git
   cd face-recognition-attendance
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Flask Server**
   ```sh
   python finaljoined.py
   ```
4. **Access the Web Interface**
   Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Loads the main interface |
| `/capture` | POST | Captures student face images |
| `/attendance` | GET | Opens the attendance capture interface |
| `/train_model` | POST | Trains the face recognition model |
| `/video_feed` | GET | Streams live face capture |
| `/video_feed_attendence` | GET | Streams live attendance recognition |
| `/capture_attendance` | POST | Recognizes faces and logs attendance |
| `/table` | GET | Displays attendance records |

## Project Workflow
1. **Add a Student**: Capture student images and store them in the dataset.
2. **Train the Model**: Fine-tune MobileNetV2 on the dataset.
3. **Recognize Faces**: Live face detection and recognition.
4. **Log Attendance**: Store recognized student attendance in a CSV file.
5. **View Attendance**: Display attendance records in tabular form.

## Future Enhancements
- **Improve Model Accuracy** with more training data.
- **Integrate with Student Databases** for automated record management.
- **Develop a Mobile App** for attendance tracking.
- **Cloud Integration** for remote storage and real-time syncing.

## License
This project is licensed under the **MIT License**.

## Contributors
- [Your Name](https://github.com/your-username)

---
### 🚀 Contributions Welcome!
Feel free to fork this repository, improve the project, and submit a pull request. Let's build an efficient and intelligent attendance system together!
