from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import os
import datetime
import asyncio
import time
from color_utils import extract_color
from audio_utils import WarningPlayer

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/face_cigar_1.pt")
    smoke_vapepod_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/smoke_vapepod_1.pt")
    clothing_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/src/models/clothing.pt")
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
    last_warning_time = None
    warning_player = WarningPlayer()

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

            # 담배 및 얼굴 탐지
            for box in results_face_cigar[0].boxes:
                class_name = face_cigar_model.names[int(box.cls)]
                if class_name == "cigarette":
                    cigar_detected = True
                    detected_class_names.append(class_name)
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
                elif class_name == "face":
                    person_detected = True
                    detected_class_names.append(class_name)

            # 전자담배 탐지
            for box in results_smoke_vapepod[0].boxes:
                class_name = smoke_vapepod_model.names[int(box.cls)]
                if class_name == "vapepod":
                    cigar_detected = True
                    detected_class_names.append(class_name)
                    xyxy = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(
                        frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        class_name,
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

            # 담배나 전자담배를 피우고 있는 사람 탐지 시 경고
            if cigar_detected and person_detected:
                clothing_detected = False
                clothing_color = None
                clothing_class_name = None

                # 옷 탐지 및 색상 추출
                for box in results_clothing[0].boxes:
                    class_name = clothing_model.names[int(box.cls)]
                    if class_name in [
                        "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear", 
                        "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", 
                        "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
                    ]:
                        clothing_color = extract_color(frame, box.xyxy[0])
                        clothing_class_name = class_name
                        clothing_detected = True
                        break

                # 경고 음성 출력
                if clothing_detected and clothing_color:
                    warning_player.play_warning(detected_class_names, clothing_color, clothing_class_name)
                else:
                    warning_player.play_warning(detected_class_names)

            # 20초 이상 동일한 사람이 담배를 피우고 있으면 다시 경고
            current_time = time.time()
            if cigar_detected and person_detected:
                if last_warning_time is None or current_time - last_warning_time > 20:
                    last_warning_time = current_time

            # 탐지 결과 화면 출력
            rendered_frame = results_face_cigar[0].plot()

            # 저장된 이미지에 탐지된 객체를 포함하여 저장
            capture_path = f"static/capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(capture_path, frame)

            # WebSocket 데이터 전송
            detected_objects = [
                face_cigar_model.names[int(box.cls)] for box in results_face_cigar[0].boxes
            ]
            await websocket.send_json({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": detected_objects,
                "image_url": f"/static/{os.path.basename(capture_path)}"
            })

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
