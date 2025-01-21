import numpy as np

# 색상 기준 정의
COLOR_RANGES = {
    "파란색": lambda b, g, r: b > 150 and g < 100 and r < 100,
    "밝은 파란색": lambda b, g, r: b > 200 and g < 150 and r < 150,
    "어두운 파란색": lambda b, g, r: 100 < b < 150 and g < 50 and r < 50,
    "빨간색": lambda b, g, r: b < 100 and g < 100 and r > 150,
    "밝은 빨간색": lambda b, g, r: b < 100 and g < 100 and r > 200,
    "어두운 빨간색": lambda b, g, r: b < 50 and g < 50 and r > 100,
    "초록색": lambda b, g, r: b < 100 and g > 150 and r < 100,
    "밝은 초록색": lambda b, g, r: b < 150 and g > 200 and r < 150,
    "어두운 초록색": lambda b, g, r: b < 50 and g > 100 and r < 50,
    "노란색": lambda b, g, r: b < 150 and g > 150 and r > 150,
    "밝은 노란색": lambda b, g, r: b < 100 and g > 200 and r > 200,
    "어두운 노란색": lambda b, g, r: b < 50 and g > 100 and r > 100,
    "회색": lambda b, g, r: 80 < b < 120 and 80 < g < 120 and 80 < r < 120,
    "밝은 회색": lambda b, g, r: 120 < b < 180 and 120 < g < 180 and 120 < r < 180,
    "어두운 회색": lambda b, g, r: 50 < b < 80 and 50 < g < 80 and 50 < r < 80,
    "검은색": lambda b, g, r: b < 50 and g < 50 and r < 50,
    "흰색": lambda b, g, r: b > 200 and g > 200 and r > 200,
    "핑크색": lambda b, g, r: b < 100 and g < 100 and 150 < r < 200,
    "밝은 핑크색": lambda b, g, r: b < 150 and g < 150 and r > 200,
    "어두운 핑크색": lambda b, g, r: b < 50 and g < 50 and 100 < r < 150,
    "보라색": lambda b, g, r: b > 100 and g < 100 and r > 100,
    "밝은 보라색": lambda b, g, r: b > 150 and g < 150 and r > 150,
    "어두운 보라색": lambda b, g, r: b > 50 and g < 50 and r > 50,
    "갈색": lambda b, g, r: 60 < b < 120 and 40 < g < 80 and 20 < r < 60,
    "밝은 갈색": lambda b, g, r: 100 < b < 150 and 80 < g < 100 and 50 < r < 80,
    "어두운 갈색": lambda b, g, r: 40 < b < 80 and 30 < g < 50 and 10 < r < 40,
    "주황색": lambda b, g, r: b < 100 and 80 < g < 150 and r > 150,
    "밝은 주황색": lambda b, g, r: b < 80 and 100 < g < 180 and r > 200,
    "어두운 주황색": lambda b, g, r: b < 50 and 50 < g < 100 and 100 < r < 150,
    "베이지색": lambda b, g, r: 150 < b < 200 and 200 < g < 250 and 170 < r < 220,
    "밝은 베이지색": lambda b, g, r: 180 < b < 220 and 220 < g < 255 and 200 < r < 250,
    "어두운 베이지색": lambda b, g, r: 120 < b < 150 and 150 < g < 200 and 100 < r < 170
}

# 색 추출 함수
def extract_color(image, box):
    x1, y1, x2, y2 = map(int, box)  # 좌표로 변환
    cropped_image = image[y1:y2, x1:x2]
    average_color = np.mean(cropped_image, axis=(0, 1))  # 평균 BGR 값
    b, g, r = average_color  # 평균 색상 분리

    # 색상 분류
    for color_name, condition in COLOR_RANGES.items():
        if condition(b, g, r):
            return color_name

    return "알 수 없는 색상의"  # 정의된 조건에 맞지 않으면 기본값 반환
