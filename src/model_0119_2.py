from ultralytics import YOLO
import cv2
import time
from color_utils import extract_color
from audio_utils import WarningPlayer

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/face_cigar_1.pt")
    smoke_vapepod_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/smoke_vapepod_1.pt")
    clothing_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/clothing.pt")
    return face_cigar_model, smoke_vapepod_model, clothing_model

# 메인 함수
def main():
    face_cigar_model, smoke_vapepod_model, clothing_model = init_models()
    warning_player = WarningPlayer()  # 경고 음성 재생 관리 객체

    # 웹캠 열기
    capture = cv2.VideoCapture(0)
    time.sleep(2)  # 웹캠 안정화 대기
    if not capture.isOpened():
        print("웹캠 연결 오류")
        return

    last_warning_time = None

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("영상 캡처 실패")
                break

            # YOLO 모델로 객체 탐지
            results_face_cigar = face_cigar_model(frame)
            results_smoke_vapepod = smoke_vapepod_model(frame)
            results_clothing = clothing_model(frame)

            cigar_detected = False
            person_detected = False
            detected_class_names = []

            # 담배 및 얼굴 탐지
            for result in results_face_cigar:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "cigarette":
                        cigar_detected = True
                        detected_class_names.append(class_name)
                    elif class_name == "face":
                        person_detected = True
                        detected_class_names.append(class_name)

            # 전자담배 탐지
            for result in results_smoke_vapepod:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "vapepod":
                        cigar_detected = True
                        detected_class_names.append(class_name)
                    elif class_name == "smoke":
                        detected_class_names.append(class_name)

            # 담배나 전자담배를 피우고 있는 사람 탐지 시 경고
            if cigar_detected and person_detected:
                clothing_detected = False
                clothing_color = None
                clothing_class_name = None

                # 옷 탐지 및 색상 추출
                for result in results_clothing:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]

                        if class_name in [
                            "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear", 
                            "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", 
                            "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
                        ]:
                            clothing_color = extract_color(frame, box.xyxy[0])  # 옷 색상 추출
                            clothing_class_name = class_name
                            clothing_detected = True
                            break
                    if clothing_detected:
                        break

                # 경고 음성 출력
                if clothing_detected and clothing_color:
                    warning_player.play_warning(detected_class_names, clothing_color, clothing_class_name)
                else:
                    warning_player.play_warning(detected_class_names)

            # 탐지 결과 화면 출력
            rendered_frame = results_face_cigar[0].plot()  # 탐지 결과 시각화
            cv2.imshow("Cigarette/Vapepod Detection", rendered_frame)

            # 20초 이상 동일한 사람이 담배를 피우고 있으면 다시 경고
            if cigar_detected and person_detected:
                current_time = time.time()
                if last_warning_time is None or current_time - last_warning_time > 20:
                    last_warning_time = current_time

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 리소스 해제
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
