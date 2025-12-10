import sys
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

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

# ============ Thread للكاميرا ============
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, str, float, float)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
        # إعداد MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
        
        # تحميل الموديل
        self.model = DrowsinessCNN()
        self.model.load_state_dict(torch.load('./saved_model/open_cloded_eye_detector.pth', 
                                               map_location='cpu'))
        self.model.eval()
        
        # التحويلات
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.input_transform = transforms.Compose([
            transforms.Resize((145, 145)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        
        self.cap = None
    
    def get_eye_region(self, landmarks, eye_indexes, frame_shape):
        h, w = frame_shape[:2]
        points = np.array([[int(landmarks[idx].x*w), int(landmarks[idx].y*h)] 
                          for idx in eye_indexes])
        x_min = max(0, np.min(points[:,0]) - 20)
        x_max = min(w, np.max(points[:,0]) + 20)
        y_min = max(0, np.min(points[:,1]) - 15)
        y_max = min(h, np.max(points[:,1]) + 15)
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def predict_eye(self, eye_image):
        if eye_image.size == 0:
            return 0.5
        eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
        eye_pil = Image.fromarray(eye_rgb)
        eye_tensor = self.input_transform(eye_pil).unsqueeze(0)
        with torch.no_grad():
            output = self.model(eye_tensor)
            prob = torch.sigmoid(output).item()
        return prob
    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                status = "No Face"
                right_pred = 0.0
                left_pred = 0.0
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        rx1, ry1, rx2, ry2 = self.get_eye_region(landmarks, 
                                                                  self.RIGHT_EYE_INDEXES, 
                                                                  frame.shape)
                        lx1, ly1, lx2, ly2 = self.get_eye_region(landmarks, 
                                                                  self.LEFT_EYE_INDEXES, 
                                                                  frame.shape)
                        
                        right_eye_img = frame[ry1:ry2, rx1:rx2]
                        left_eye_img = frame[ly1:ly2, lx1:lx2]
                        
                        right_pred = self.predict_eye(right_eye_img)
                        left_pred = self.predict_eye(left_eye_img)
                        
                        avg_score = (right_pred + left_pred) / 2
                        status = "AWAKE" if avg_score >= 0.5 else "SLEEPY"
                        
                        color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
                        
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 2)
                
                self.frame_ready.emit(frame, status, right_pred, left_pred)
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

# ============ النافذة الرئيسية ============
class DrowsinessDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("نظام كشف النعاس - Drowsiness Detection System")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:pressed {
                background-color: #89dceb;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QGroupBox {
                border: 2px solid #45475a;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                color: #cdd6f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        self.camera_thread = None
        self.is_running = False
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # الجانب الأيسر - الفيديو
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 10px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("الكاميرا متوقفة\nCamera Stopped")
        
        left_layout.addWidget(self.video_label)
        
        # أزرار التحكم
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ تشغيل الكاميرا\nStart Camera")
        self.start_btn.clicked.connect(self.start_camera)
        
        self.stop_btn = QPushButton("⏸ إيقاف الكاميرا\nStop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        # الجانب الأيمن - المعلومات
        right_layout = QVBoxLayout()
        
        # حالة السائق
        status_group = QGroupBox("حالة السائق - Driver Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("⚪ No Face Detected")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #313244;
                padding: 20px;
                border-radius: 10px;
                color: #cdd6f4;
            }
        """)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        
        # معلومات العيون
        eyes_group = QGroupBox("تفاصيل العيون - Eyes Details")
        eyes_layout = QVBoxLayout()
        
        self.right_eye_label = QLabel("العين اليمنى | Right Eye: --")
        self.right_eye_label.setFont(QFont("Arial", 12))
        
        self.left_eye_label = QLabel("العين اليسرى | Left Eye: --")
        self.left_eye_label.setFont(QFont("Arial", 12))
        
        self.avg_label = QLabel("المتوسط | Average: --")
        self.avg_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        eyes_layout.addWidget(self.right_eye_label)
        eyes_layout.addWidget(self.left_eye_label)
        eyes_layout.addWidget(self.avg_label)
        eyes_group.setLayout(eyes_layout)
        
        # معلومات النظام
        info_group = QGroupBox("معلومات النظام - System Info")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "📌 هذا النظام يستخدم الذكاء الاصطناعي لكشف النعاس\n"
            "📌 This system uses AI to detect drowsiness\n\n"
            "✓ يتم تحليل حركة العيون في الوقت الفعلي\n"
            "✓ Real-time eye movement analysis\n\n"
            "⚠ عند اكتشاف النعاس، سيتغير اللون إلى أحمر\n"
            "⚠ Color changes to red when drowsiness detected"
        )
        info_text.setWordWrap(True)
        info_text.setFont(QFont("Arial", 10))
        info_text.setStyleSheet("QLabel { padding: 10px; }")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        
        right_layout.addWidget(status_group)
        right_layout.addWidget(eyes_group)
        right_layout.addWidget(info_group)
        right_layout.addStretch()
        
        # إضافة التخطيطات للنافذة الرئيسية
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
    
    def start_camera(self):
        if not self.is_running:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.running = True
            self.camera_thread.start()
            
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
    
    def stop_camera(self):
        if self.is_running and self.camera_thread:
            self.camera_thread.stop()
            self.is_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.video_label.setText("الكاميرا متوقفة\nCamera Stopped")
            self.status_label.setText("⚪ No Face Detected")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #313244;
                    padding: 20px;
                    border-radius: 10px;
                    color: #cdd6f4;
                }
            """)
    
    def update_frame(self, frame, status, right_pred, left_pred):
        # تحويل الإطار لعرضه في Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.KeepAspectRatio
        ))
        
        # تحديث الحالة
        if status == "AWAKE":
            self.status_label.setText("✓ مستيقظ - AWAKE")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #a6e3a1;
                    color: #1e1e2e;
                    padding: 20px;
                    border-radius: 10px;
                }
            """)
        elif status == "SLEEPY":
            self.status_label.setText("⚠ نعسان - SLEEPY")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f38ba8;
                    color: #1e1e2e;
                    padding: 20px;
                    border-radius: 10px;
                }
            """)
        else:
            self.status_label.setText("⚪ No Face Detected")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #313244;
                    padding: 20px;
                    border-radius: 10px;
                    color: #cdd6f4;
                }
            """)
        
        # تحديث معلومات العيون
        if status != "No Face":
            avg = (right_pred + left_pred) / 2
            self.right_eye_label.setText(f"العين اليمنى | Right Eye: {right_pred:.2%}")
            self.left_eye_label.setText(f"العين اليسرى | Left Eye: {left_pred:.2%}")
            self.avg_label.setText(f"المتوسط | Average: {avg:.2%}")
        else:
            self.right_eye_label.setText("العين اليمنى | Right Eye: --")
            self.left_eye_label.setText("العين اليسرى | Left Eye: --")
            self.avg_label.setText("المتوسط | Average: --")
    
    def closeEvent(self, event):
        if self.is_running:
            self.stop_camera()
        event.accept()

# ============ تشغيل التطبيق ============
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrowsinessDetectorGUI()
    window.show()
    sys.exit(app.exec_())