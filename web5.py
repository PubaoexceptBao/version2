from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import datetime
import asyncio
import numpy as np
import mediapipe as mp
from color_utils import extract_color
from audio_utils import WarningPlayer

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands  # 손 감지 추가

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/face_cigar_1.pt")
    smoke_vapepod_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/smoke_vapepod_1.pt")
    clothing_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/clothing.pt")
    return face_cigar_model, smoke_vapepod_model, clothing_model

# 팔 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# FastAPI 앱 생성
app = FastAPI()

# Static 폴더 생성 (이미지 저장용)
if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates 폴더 생성
if not os.path.exists("templates"):
    os.mkdir("templates")

# HTML 파일 경로
html_file_path = "templates/main.html"

# HTML 반환
@app.get("/")
async def get():
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# WebSocket 경로
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    face_cigar_model, smoke_vapepod_model, clothing_model = init_models()
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    warning_player = WarningPlayer()

    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 탐지 수행
            results_face_cigar = face_cigar_model.predict(frame)
            results_smoke_vapepod = smoke_vapepod_model.predict(frame)
            results_clothing = clothing_model.predict(frame)

            cigar_detected = False
            person_detected = False
            detected_class_names = []

            for box in results_face_cigar[0].boxes:
                class_name = face_cigar_model.names[int(box.cls)]
                if class_name == "cigarette":
                    cigar_detected = True
                    detected_class_names.append(class_name)
                    cv2.rectangle(
                        frame,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                        (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        class_name,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                elif class_name == "face":
                    person_detected = True
                    detected_class_names.append(class_name)

            # Mediapipe Hands 처리
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hands.process(image_rgb)

            # Mediapipe Pose 처리
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                landmarks = results.pose_landmarks.landmark
                
                try:
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if left_angle > 160:
                        left_stage = "down"
                    if left_angle < 30 and left_stage == "down":
                        left_stage = "up"
                        left_counter += 1

                    if right_angle > 160:
                        right_stage = "down"
                    if right_angle < 30 and right_stage == "down":
                        right_stage = "up"
                        right_counter += 1

                    # 화면에 Reps 표시
                    cv2.putText(
                        frame, f"Left Reps: {left_counter}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, f"Right Reps: {right_counter}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
                    )
                except:
                    pass

            # 경고 조건
            if cigar_detected and person_detected and (left_counter >= 2 or right_counter >= 2):
                warning_player.play_warning(detected_class_names)
                left_counter = 0
                right_counter = 0

            # 탐지 결과 저장 및 WebSocket 전송
            capture_path = f"static/capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            annotated_frame = frame.copy()

            # 저장된 이미지에 랜드마크와 Reps 포함
            cv2.putText(
                annotated_frame, f"Left Reps: {left_counter}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated_frame, f"Right Reps: {right_counter}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
            )
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            cv2.imwrite(capture_path, annotated_frame)
            
            await websocket.send_json({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": detected_class_names,
                "image_url": f"/static/{os.path.basename(capture_path)}"
            })

            # 화면 출력
            cv2.imshow("Cigarette and Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
