from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import datetime
import asyncio

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/face_cigar_1.pt")
    smoke_vapepod_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/smoke_vapepod_1.pt")
    clothing_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/clothing.pt")
    return face_cigar_model, smoke_vapepod_model, clothing_model

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
    cap = cv2.VideoCapture(0)  # 웹캠

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 탐지 수행
            results = face_cigar_model.predict(frame)
            detected_objects = []

            for box in results[0].boxes:
                class_name = face_cigar_model.names[int(box.cls)]
                detected_objects.append(class_name)

                if class_name == "cigarette":
                    xyxy = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(
                        frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        class_name,
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # 현재 시간과 날짜
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 이미지 저장
            capture_path = f"static/capture_{current_time.replace(':', '-').replace(' ', '_')}.jpg"
            cv2.imwrite(capture_path, frame)

            # WebSocket으로 결과 전송
            await websocket.send_json({
                "time": current_time,
                "objects": detected_objects,
                "image_url": f"/static/{os.path.basename(capture_path)}"
            })

            await asyncio.sleep(1)
    finally:
        cap.release()
