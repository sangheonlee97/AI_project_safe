import cv2
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model

# 비디오 파일 경로 설정
video_path = "C:\\safezone\\data\\temp\\접근알림.mp4"
# video_path = "C:\\safezone\\data\\temp\\깊이정보.mp4"
output_path = "C:\\safezone\\data\\temp\\result1.mp4"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오 저장 객체 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 프레임 속도 가져오기

slow_fps = fps / 2  # 비디오 속도를 절반으로 줄이기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, slow_fps, (width, height))

# 총 50개의 클래스에 대한 고유한 색상을 생성
np.random.seed(0)  # 시드 고정
num_classes = 100
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

# 객체 클래스 ID 정의
person_class_id = 0  # 예: 'person' 클래스 ID
chair_class_id = 56  # 예: 'chair' 클래스 ID

# 거리 임계값 설정 (예: 픽셀 단위)
distance_threshold = 200

# 한글 폰트 설정 (NanumGothic.ttf 파일 경로를 지정하세요)
font_path = "C:\\Windows\\Fonts\\malgun.ttf"  # 폰트 파일 경로 설정
font = ImageFont.truetype(font_path, 24)

# 객체 감지 및 주석 추가
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 모델에 전달하여 예측 수행
    results = model(frame)

    # 예측 결과에서 바운딩 박스 정보 추출
    predictions = results[0].boxes if results else []

    # OpenCV 이미지를 Pillow 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 객체 좌표 저장
    persons = []
    chairs = []

    # 프레임에 주석 추가 및 객체 좌표 저장
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred.xyxy[0])  # Get bounding box coordinates
        conf = pred.conf[0]  # Confidence
        cls = int(pred.cls[0])  # Class
        label = f"{model.names[cls]} {conf:.2f}"
        color = colors[cls].tolist()  # 클래스에 해당하는 색상 가져오기
        draw.rectangle(((x1, y1), (x2, y2)), outline=tuple(color), width=2)
        draw.text((x1, y1 - 10), label, font=font, fill=tuple(color) + (255,))

        # 객체 중심 좌표 계산
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # 객체를 클래스에 따라 분류
        if cls == person_class_id:
            persons.append((center_x, center_y))
        elif cls == chair_class_id:
            chairs.append((center_x, center_y))

    # 객체 간 거리 계산 및 알림
    for px, py in persons:
        for cx, cy in chairs:
            distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if distance < distance_threshold:
                print("경고: 사람 객체가 의자 객체에 접근했습니다!")
                draw.text((10, 10), "경고: 사람 객체가 의자 객체에 접근했습니다!", font=font, fill=(255, 0, 0, 255))

    # Pillow 이미지를 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 주석이 추가된 프레임을 비디오 파일에 저장
    out.write(frame)

    # 주석이 추가된 프레임을 디스플레이 (옵션)
    cv2.imshow('frame', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
