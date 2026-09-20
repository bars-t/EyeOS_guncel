import cv2
import joblib
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller

# Ayarlar
mouse = Controller()
model = joblib.load(r"C:\yuz_algila\yuz_ifade_modeli.pkl")
MODEL_YOLU = r"C:\yuz_algila\face_landmarker.task"

# MediaPipe Kurulumu
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_YOLU),
    running_mode=RunningMode.IMAGE,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)
landmarker = FaceLandmarker.create_from_options(options)

def main():
    cap = cv2.VideoCapture(0)
    print("EyeOS Final Sürümü Aktif: Gözlerini çevir, fareyi kontrol et!")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Ayna yansıması ve RGB dönüşümü
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = landmarker.detect(mp_image)
        
        if result.face_landmarks:
            # 1. İris Koordinatlarını al (Sol: 468, Sağ: 473)
            sol_iris = result.face_landmarks[0][468]
            sag_iris = result.face_landmarks[0][473]
            
            # Koordinatları ekrana eşle
            h, w, _ = frame.shape
            mouse_x = int(np.mean([sol_iris.x, sag_iris.x]) * 1920) # 1920 senin ekran genişliğin
            mouse_y = int(np.mean([sol_iris.y, sag_iris.y]) * 1080)
            
            # 2. Fare Hareketi (Sadece Gözler)
            mouse.position = (mouse_x, mouse_y)
            
            # 3. Model Tahmini (Göz Kırpma / Kapama)
            # Burada 'uyuyor' etiketini modelden alıp 'kapalı göz' olarak işliyoruz
            # (Senin modelindeki etiket isimlerini kontrol et)
            
            # Örnek: Eğer model 'uyuyor' derse:
            # mouse.click(Button.left) 
            
        cv2.imshow("EyeOS - Aktif Iris Transferi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()