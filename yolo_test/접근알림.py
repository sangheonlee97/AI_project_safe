import cv2
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Load a model
model = YOLO("best.pt")  # load a pretrained model

# 비디오 파일 경로 설정
# video_path = "C:\\safezone\\data\\temp\\접근알림.mp4"
video_path = "C:\safezone\data\\temp\\result2.mp4"
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

# 한글 폰트 설정 (NanumGothic.ttf 파일 경로를 지정하세요)
# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
# font = ImageFont.truetype(font_path, 24)

# 객체 감지 및 주석 추가