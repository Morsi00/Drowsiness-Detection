from flask import Flask, render_template, Response, jsonify
# from flask_cors import CORS
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)
# CORS(app)

# ============ موديل كشف النعاس ============
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9 * 9 * 512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), -1)
        return self.dense(x)

# ============ إعداد MediaPipe ============
mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

# ============ تحميل الموديل ============
model = DrowsinessCNN()
model.load_state_dict(torch.load('./saved_model/open_cloded_eye_detector.pth', map_location='cpu'))
model.eval()

# ============ التحويلات ============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
input_transform = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ============ متغيرات عامة ============
camera = None
face_mesh = None
current_status = {
    'status': 'No Face',
    'right_eye_score': 0.0,
    'left_eye_score': 0.0,
    'avg_score': 0.0,
    'right_eye_region': None,
    'left_eye_region': None
}

# ======= دوال مساعدة =======
def get_eye_region(landmarks, eye_indexes, frame_shape):
    h, w = frame_shape[:2]
    points = np.array([[int(landmarks[idx].x*w), int(landmarks[idx].y*h)] for idx in eye_indexes])
    x_min = max(0, np.min(points[:,0]) - 20)
    x_max = min(w, np.max(points[:,0]) + 20)
    y_min = max(0, np.min(points[:,1]) - 15)
    y_max = min(h, np.max(points[:,1]) + 15)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def predict_eye(eye_image):
    if eye_image.size == 0:
        return 0.5
    eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    eye_pil = Image.fromarray(eye_rgb)
    eye_tensor = input_transform(eye_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(eye_tensor)
        prob = torch.sigmoid(output).item()
    return prob

def generate_frames():
    global camera, face_mesh, current_status
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        status = "No Face"
        right_pred = 0.0
        left_pred = 0.0
        right_region = None
        left_region = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                rx1, ry1, rx2, ry2 = get_eye_region(landmarks, RIGHT_EYE_INDEXES, frame.shape)
                lx1, ly1, lx2, ly2 = get_eye_region(landmarks, LEFT_EYE_INDEXES, frame.shape)
                
                right_eye_img = frame[ry1:ry2, rx1:rx2]
                left_eye_img = frame[ly1:ly2, lx1:lx2]
                
                right_pred = predict_eye(right_eye_img)
                left_pred = predict_eye(left_eye_img)
                
                avg_score = (right_pred + left_pred) / 2
                status = "AWAKE" if avg_score >= 0.5 else "SLEEPY"
                
                color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
                
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 2)
                cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                right_region = {'x': rx1, 'y': ry1, 'width': rx2-rx1, 'height': ry2-ry1}
                left_region = {'x': lx1, 'y': ly1, 'width': lx2-lx1, 'height': ly2-ly1}
        
        current_status = {
            'status': status,
            'right_eye_score': float(right_pred),
            'left_eye_score': float(left_pred),
            'avg_score': float((right_pred + left_pred) / 2),
            'right_eye_region': right_region,
            'left_eye_region': left_region
        }
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ======= Routes =======
@app.route('/')
def homepage():
    """صفحة الترحيب"""
    return render_template('homepage.html')

@app.route('/drowsiness')
def drowsiness_page():
    """صفحة كشف النعاس"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(current_status)

@app.route('/stop_camera')
def stop_camera():
    global camera, face_mesh
    if camera:
        camera.release()
        camera = None
    if face_mesh:
        face_mesh.close()
        face_mesh = None
    return jsonify({'message': 'Camera stopped'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
