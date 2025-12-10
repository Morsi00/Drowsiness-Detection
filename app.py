"""
Flask App for Live Drowsiness Detection
Using MediaPipe + Existing CNN Model
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import deque

# from cnn_model import DrowsinessCNN  # Import your CNN model

app = Flask(__name__)

# ============ Load Model ============
print("📦 Loading model...")
# model = DrowsinessCNN()
model.load_state_dict(torch.load('./saved_model/drowsiness.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded successfully!")

# ============ Setup MediaPipe ============
print("📦 Loading MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe loaded successfully!")

# ============ Settings ============
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0),(1,1,1))
])

RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

drowsiness_scores = deque(maxlen=300)
frame_count = 0
current_status = {
    "status": "Analyzing...",
    "status_en": "Analyzing...",
    "score": 0.0,
    "samples": 0,
    "alert": False
}

# ============ Helper Functions ============
def get_eye_region(landmarks, eye_indexes, frame_shape):
    h, w = frame_shape[:2]
    points = []

    for idx in eye_indexes:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])

    points = np.array(points)
    x_min = max(0, np.min(points[:,0]) - 20)
    x_max = min(w, np.max(points[:,0]) + 20)
    y_min = max(0, np.min(points[:,1]) - 15)
    y_max = min(h, np.max(points[:,1]) + 15)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def predict_eye(eye_image):
    if eye_image is None or eye_image.size == 0:
        return None
    try:
        eye_pil = Image.fromarray(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB))
        eye_resized = eye_pil.resize((145,145))
        eye_tensor = input_transform(eye_resized).unsqueeze(0)
        with torch.no_grad():
            output = model(eye_tensor)
            prob = torch.sigmoid(output).item()
        return prob
    except:
        return None

# ============ Video Generator ============
def generate_frames():
    global frame_count, drowsiness_scores, current_status
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return

    print("✅ Live streaming started...")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            rx1, ry1, rx2, ry2 = get_eye_region(landmarks, RIGHT_EYE_INDEXES, frame.shape)
            lx1, ly1, lx2, ly2 = get_eye_region(landmarks, LEFT_EYE_INDEXES, frame.shape)

            right_eye_img = frame[ry1:ry2, rx1:rx2]
            left_eye_img = frame[ly1:ly2, lx1:lx2]

            # Draw rectangles
            cv2.rectangle(frame, (rx1,ry1),(rx2,ry2),(0,255,0),2)
            cv2.rectangle(frame, (lx1,ly1),(lx2,ly2),(0,255,0),2)
            cv2.putText(frame,"R",(rx1, ry1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv2.putText(frame,"L",(lx1, ly1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            predictions = []
            for eye_img in [right_eye_img, left_eye_img]:
                pred = predict_eye(eye_img)
                if pred is not None:
                    predictions.append(pred)

            if predictions:
                avg_score = sum(predictions)/len(predictions)
                drowsiness_scores.append(avg_score)
                if len(drowsiness_scores)>=30:
                    mean_score = sum(drowsiness_scores)/len(drowsiness_scores)
                    if mean_score < 0.5:
                        status_ar = "Awake"
                        emoji = "😊"
                        color = (0,255,0)
                        alert = False
                    else:
                        status_ar = "Sleepy"
                        emoji = "😴"
                        color = (0,0,255)
                        alert = True
                    current_status.update({
                        "status": f"{emoji} {status_ar}",
                        "status_en": status_ar,
                        "score": float(mean_score),
                        "samples": len(drowsiness_scores),
                        "alert": alert
                    })
                else:
                    status_ar = "Analyzing..."
                    emoji = "⏳"
                    color = (255,255,0)
                cv2.putText(frame,f"{emoji} {status_ar}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
                cv2.putText(frame,f"Score: {avg_score:.3f}",(10,85),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            cv2.putText(frame,"No Face Detected",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        ret, buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

    cap.release()

# ============ Flask Routes ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(current_status)

@app.route('/reset')
def reset():
    global drowsiness_scores, frame_count
    drowsiness_scores.clear()
    frame_count = 0
    return jsonify({"message":"Reset successfully!"})

# ============ Run App ============
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 Drowsiness Detection System - Flask Web App")
    print("="*60)
    print("📡 Server running at: http://localhost:5000")
    print("🌐 Access from network: http://YOUR_IP:5000")
    print("⚠️ Press Ctrl+C to stop")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
