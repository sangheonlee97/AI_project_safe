from roboflow import Roboflow
import supervision as sv
import cv2

# Roboflow API 키와 프로젝트 설정
rf = Roboflow(api_key="OanmvFxqS7uMTuji06li")
project = rf.workspace().project("ppe1-zzzjz")
model = project.version(4).model

# 동영상 파일 열기
video = cv2.VideoCapture("C:\yoloProject\data\\KakaoTalk_20240603_115441893.mp4")

# 동영상의 각 프레임을 처리
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # 모델을 사용해 프레임 분석
    result = model.predict(frame, confidence=40, overlap=30).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 검출된 객체에 대한 주석 추가
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    # 주석이 추가된 프레임 표시
    cv2.imshow("Annotated Frame", annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 자원 해제
video.release()
cv2.destroyAllWindows()