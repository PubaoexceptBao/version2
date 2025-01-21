from gtts import gTTS
import time
import subprocess

class WarningPlayer:
    def __init__(self):
        self.last_warning_time = None  # 마지막 경고 시간을 기록

    def get_clothing_type(self, clothing_class_name):
        """옷 클래스 이름에 따라 적절한 텍스트를 반환합니다."""
        if clothing_class_name in ["short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear", "long_sleeved_outwear", "vest", "sling"]:
            return "상의"
        elif clothing_class_name in ["shorts", "trousers", "skirt"]:
            return "하의"
        elif clothing_class_name in ["short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"]:
            return "옷"
        return None

    def play_warning(self, class_names, color=None, clothing_class_name=None):
        current_time = time.time()

        # 마지막 경고로부터 20초가 지나지 않았다면 새로운 경고를 무시 # 대신 경고 멘트 20초 이하이어야 함
        if self.last_warning_time and current_time - self.last_warning_time < 20:
            return

        self.last_warning_time = current_time  # 경고 시간 갱신

        # 옷 타입 결정
        clothing_type = self.get_clothing_type(clothing_class_name) if clothing_class_name else None

        # 경고 텍스트 생성
        # f"{', '.join(class_names)} 객체가 탐지되었습니다. "
        if color and clothing_type:
            warning_text = (
                f"{color} {clothing_type}을 입은 분! 이곳에서 담배를 피면 안됩니다. 다시 한 번 알려드립니다. "
                f"{color} {clothing_type}을 입은 분! 이곳에서 담배를 피면 안됩니다."
            )
        else:
            warning_text = (
                "이곳에서 담배를 피면 안됩니다. 다시 한 번 알려드립니다. 이곳에서 담배를 피면 안됩니다."
            )

        # 음성 파일 생성
        try:
            tts = gTTS(text=warning_text, lang='ko')
            warning_file = "assets\warning.mp3"
            tts.save(warning_file)

            # 음성 파일 재생
            subprocess.run(["start", warning_file], shell=True)
        except Exception as e:
            print(f"경고 음성 생성/재생 중 오류 발생: {e}")
