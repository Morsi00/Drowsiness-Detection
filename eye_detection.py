import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ============ تعريف الموديل ============
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

# ============ فتح الكاميرا ============
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_eye_region(landmarks, eye_indexes, frame_shape):
    h, w = frame_shape[:2]
    points = np.array([[int(landmarks[idx].x*w), int(landmarks[idx].y*h)] for idx in eye_indexes])
    x_min = max(0, np.min(points[:,0]) - 20)
    x_max = min(w, np.max(points[:,0]) + 20)
    y_min = max(0, np.min(points[:,1]) - 15)
    y_max = min(h, np.max(points[:,1]) + 15)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def predict_eye(eye_image):
    eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    eye_pil = Image.fromarray(eye_rgb)
    eye_tensor = input_transform(eye_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(eye_tensor)
        prob = torch.sigmoid(output).item()
    return prob

# ============ الحلقة الرئيسية ============
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status = "No Face"

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

                # رسم المستطيلات حولحول العيون
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 2)
                cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
