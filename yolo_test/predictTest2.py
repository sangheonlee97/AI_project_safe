from roboflow import Roboflow
import supervision as sv
import cv2
import time

# Roboflow API 키와 프로젝트 설정
rf = Roboflow(api_key="OanmvFxqS7uMTuji06li")
project = rf.workspace().project("ppe1-zzzjz")
model = project.version(4).model

# 동영상 파일 열기
video = cv2.VideoCapture("C:\yoloProject\data\\KakaoTalk_20240603_115441893.mp4")

# 동영상 파일 저장을 위한 설정
output_path = "C:\yoloProject\data\\result2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
sample_rate = 1  # 샘플링 비율, 예를 들어, 5프레임마다 한 번 예측

# 동영상의 각 프레임을 처리
while True:
    ret, frame = video.read()
    if not ret:
        break

    if frame_count % sample_rate == 0:
        try:
            # 모델을 사용해 프레임 분석
            result = model.predict(frame, confidence=40, overlap=30).json()
            labels = [item["class"] for item in result["predictions"]]
            detections = sv.Detections.from_inference(result)  # from_roboflow 대신 from_inference 사용

            bounding_box_annotator = sv.BoundingBoxAnnotator()  # BoxAnnotator 대신 BoundingBoxAnnotator 사용
            label_annotator = sv.LabelAnnotator()

            # 검출된 객체에 대한 주석 추가
            annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            annotated_frame = frame  # 예측 실패 시 원본 프레임 사용
    else:
        annotated_frame = frame  # 샘플링 비율에 해당하지 않는 경우 원본 프레임 사용

    # 주석이 추가된 프레임을 동영상 파일에 저장
    out.write(annotated_frame)
    frame_count += 1

# 자원 해제
video.release()
out.release()
