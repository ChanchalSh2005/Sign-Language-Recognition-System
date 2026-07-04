import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from PIL import Image
import joblib
import streamlit as st
from gtts import gTTS
import base64

# ==========================================
# 1. PAGE SETUP & UI
# ==========================================
st.set_page_config(page_title="Sign Language Translator", layout="centered", page_icon="🤟")
st.title("🤟 Real-Time Sign Language Translator")
st.write("Click a photo of your gesture to get a prediction and voice output!")

# ==========================================
# 2. PYTORCH ARCHITECTURE (33 Features)
# ==========================================
class SignMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(33, 128),  # 33 Input Features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 44)
        )
    def forward(self, x): 
        return self.model(x)

# ==========================================
# 3. CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load PyTorch Model
    model = SignMLP().to(device)
    model_path = 'sign_mlp_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
    else:
        st.error(f"❌ Model file not found! Please ensure '{model_path}' is in the folder.")
        
    # 2. Load Encoders (SCALER RESTORED FOR LOCAL USE)
    le = joblib.load('label_encoder.pkl') if os.path.exists('label_encoder.pkl') else None
    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None

    # 3. Load MediaPipe Hand Landmarker
    if not os.path.exists('hand_landmarker.task'):
        os.system('wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), 
        num_hands=2
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    
    # 4. Load MediaPipe Pose Landmarker
    if not os.path.exists('pose_landmarker_lite.task'):
        os.system('wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task')
    pose_options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    
    return model, le, scaler, hand_detector, pose_detector, device

model, le, scaler, hand_detector, pose_detector, device = load_models()

# ==========================================
# 4. GEOMETRIC MATH & AUDIO FUNCTIONS
# ==========================================
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def extract_robust_features(landmarks):
    features = []
    tips = [4, 8, 12, 16, 20]
    for tip in tips: features.append(calculate_distance(landmarks[0], landmarks[tip]))
    features.append(calculate_distance(landmarks[4], landmarks[8]))
    features.append(calculate_distance(landmarks[8], landmarks[12]))
    features.append(calculate_distance(landmarks[12], landmarks[16]))
    features.append(calculate_distance(landmarks[16], landmarks[20]))
    features.append(calculate_distance(landmarks[1], landmarks[4]))
    features.append(calculate_distance(landmarks[5], landmarks[8]))
    features.append(calculate_distance(landmarks[9], landmarks[12]))

    hand_size = calculate_distance(landmarks[0], landmarks[9]) + 1e-5
    return [f / hand_size for f in features] 

def play_audio(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    with open("output.mp3", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# ==========================================
# 5. CAMERA INTERFACE & INFERENCE LOOP
# ==========================================
camera_image = st.camera_input("Make a sign and click to capture!")

if camera_image is not None:
    # --- Local-Safe Image Processing (Restored) ---
    pil_img = Image.open(camera_image).convert('RGB')
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img))
    
    # Run BOTH detectors
    hand_results = hand_detector.detect(mp_image)
    pose_results = pose_detector.detect(mp_image)

    if not hand_results.hand_landmarks:
        st.warning("⚠️ No hands detected. Please make sure your hands are clearly in the frame.")
    else:
        row_data = []

        # ----------------- HAND 1 (15 Features) -----------------
        h1_wrist = hand_results.hand_landmarks[0][0]
        h1_middle = hand_results.hand_landmarks[0][9]
        row_data.append(h1_wrist.y)
        row_data.append(h1_middle.x - h1_wrist.x)
        row_data.append(h1_middle.y - h1_wrist.y)
        row_data.extend(extract_robust_features(hand_results.hand_landmarks[0]))
        
        # ----------------- HAND 2 (17 Features) -----------------
        if len(hand_results.hand_landmarks) > 1:
            h2_wrist = hand_results.hand_landmarks[1][0]
            h2_middle = hand_results.hand_landmarks[1][9]
            row_data.append(h2_wrist.y)
            row_data.append(h2_middle.x - h2_wrist.x)
            row_data.append(h2_middle.y - h2_wrist.y)
            row_data.extend(extract_robust_features(hand_results.hand_landmarks[1]))
            row_data.append(h2_wrist.x - h1_wrist.x)
            row_data.append(h2_wrist.y - h1_wrist.y)
        else:
            row_data.extend([0.0] * 17) 

        # ----------------- 3D DEPTH HINT (1 Feature) -----------------
        if pose_results.pose_landmarks:
            index_tip = hand_results.hand_landmarks[0][8]
            nose = pose_results.pose_landmarks[0][0]
            face_dist = math.sqrt(
                (index_tip.x - nose.x)**2 +
                (index_tip.y - nose.y)**2 +
                (index_tip.z - nose.z)**2
            )
            row_data.append(face_dist)
        else:
            row_data.append(0.0)

        # --- SHAPE GUARD: ENSURE EXACTLY 33 FEATURES ---
        if len(row_data) != 33:
            if len(row_data) < 33:
                row_data.extend([0.0] * (33 - len(row_data)))
            else:
                row_data = row_data[:33]

        # --- PYTORCH PREDICTION (SCALER RESTORED) ---
        if scaler:
            row_data_scaled = scaler.transform([row_data])
        else:
            row_data_scaled = [row_data] 
            
        input_tensor = torch.tensor(row_data_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_id = torch.max(probabilities, dim=1)

        class_id = predicted_class_id.item()
        confidence_score = confidence.item() * 100
        predicted_label = le.inverse_transform([class_id])[0] if le else f"Class ID {class_id}"

        # --- UI DISPLAY & VOICE OUTPUT ---
        st.success(f"### Predicted Gesture: **{predicted_label.upper()}**")
        st.info(f"Confidence Score: {confidence_score:.2f}%")
        
        if confidence_score > 60:
            play_audio(predicted_label)
        else:
            st.warning("Confidence is a bit low. Try adjusting your lighting or hand position!")