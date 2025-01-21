from ultralytics import YOLO
import cv2
import time
import mediapipe as mp
import numpy as np
from color_utils import extract_color
from audio_utils import WarningPlayer

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# YOLO 모델 초기화
def init_models():
    face_cigar_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/src/face_cigar_1.pt")
    smoke_vapepod_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/src/smoke_vapepod_1.pt")
    clothing_model = YOLO("C:/Users/zstep/Documents/pubao6/2025_2nd_Smarthon/models/src/clothing.pt")
    return face_cigar_model, smoke_vapepod_model, clothing_model

# 팔 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Middle
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 메인 함수
def main():
    # YOLO 초기화
    face_cigar_model, smoke_vapepod_model, clothing_model = init_models()
    warning_player = WarningPlayer()

    # Mediapipe Pose 초기화
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 팔 굽힘 카운트 변수
    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None

    # Mediapipe에서 사용할 랜드마크 연결
    arm_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
    ]

    # 웹캠 열기
    capture = cv2.VideoCapture(0)
    time.sleep(2)
    if not capture.isOpened():
        print("웹캠 연결 오류")
        return

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("영상 캡처 실패")
                break

            # YOLO로 객체 탐지
            results_face_cigar = face_cigar_model(frame)
            cigar_detected = False
            person_detected = False
            detected_class_names = []

            for result in results_face_cigar:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "cigarette":
                        cigar_detected = True
                        detected_class_names.append(class_name)
                    elif class_name == "face":
                        person_detected = True
                        detected_class_names.append(class_name)

            # Mediapipe Pose로 팔 동작 확인
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # 왼쪽 팔 랜드마크 추출
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # 오른쪽 팔 랜드마크 추출
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # 각도 계산
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # 왼쪽 팔 동작 감지
                    if left_angle > 160:
                        left_stage = "down"
                    if left_angle < 30 and left_stage == "down":
                        left_stage = "up"
                        left_counter += 1

                    # 오른쪽 팔 동작 감지
                    if right_angle > 160:
                        right_stage = "down"
                    if right_angle < 30 and right_stage == "down":
                        right_stage = "up"
                        right_counter += 1

                    # 각도와 카운트 표시
                    cv2.putText(image, f"Left Angle: {int(left_angle)}",
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Right Angle: {int(right_angle)}",
                                tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Left Reps: {left_counter}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Right Reps: {right_counter}",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                # Mediapipe에서 팔 랜드마크와 연결선만 표시
                for connection in arm_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]
                    start_point = tuple(np.multiply([start_landmark.x, start_landmark.y], [640, 480]).astype(int))
                    end_point = tuple(np.multiply([end_landmark.x, end_landmark.y], [640, 480]).astype(int))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # 연결선
                    cv2.circle(image, start_point, 4, (0, 0, 255), -1)  # 시작점
                    cv2.circle(image, end_point, 4, (0, 0, 255), -1)  # 끝점

            # 경고 조건: 담배 + 사람 + 왼쪽 또는 오른쪽 팔 동작
            if cigar_detected and person_detected and (left_counter >= 2 or right_counter >= 2):
                print("담배를 피우고 있음!")
                warning_player.play_warning(detected_class_names)
                left_counter = 0  # 경고 후 카운트 초기화
                right_counter = 0

            # YOLO 탐지 결과 시각화
            rendered_frame = results_face_cigar[0].plot()
            combined_frame = cv2.addWeighted(rendered_frame, 0.6, image, 0.4, 0)
            cv2.imshow("Cigarette and Pose Detection", combined_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
