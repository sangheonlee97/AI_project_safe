import cv2
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# OpenAI API 키 설정
# GPT-3 모델 초기화
llm = ChatOpenAI(temperature=0.5, max_tokens=2048, model_name='gpt-3.5-turbo')

# 대화 메모리 설정
memory = ConversationBufferMemory(memory_key="history")

# 프롬프트 템플릿 설정
template = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are an object detection assistant. "
        "Based on the conversation history:\n{history}\n\n"
        "User: {input}\n"
        "Assistant: Please extract the object names for detection from the user's input."
    )
)

# LLMChain 설정
chain = LLMChain(
    llm=llm,
    prompt=template,
    memory=memory
)

# Streamlit UI 설정
st.title("Object Detection Customizer")
st.write("안녕하세요! 감지하고 싶은 객체 이름을 입력해 주세요.")

# 사용자 입력 처리
user_input = st.text_input("객체 이름을 입력하세요:")

# 객체 이름 추출
detected_objects = []
if user_input:
    response = chain.run({"input": user_input})
    detected_objects = [obj.strip() for obj in response.split(",")]
    st.write("감지할 객체들:", detected_objects)

if detected_objects:
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model

    # 비디오 파일 경로 설정
    video_path = "C:\\safezone\\data\\temp\\접근알림.mp4"
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

    # 객체 클래스 ID 정의 (모델의 클래스 이름과 매칭)
    class_names = model.names

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

        # 프레임에 주석 추가 및 객체 좌표 저장
        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred.xyxy[0])  # Get bounding box coordinates
            conf = pred.conf[0]  # Confidence
            cls = int(pred.cls[0])  # Class
            label = f"{model.names[cls]} {conf:.2f}"
            color = colors[cls].tolist()  # 클래스에 해당하는 색상 가져오기

            # 입력된 객체 이름과 일치하는 경우에만 처리
            if class_names[cls] in detected_objects:
                draw.rectangle(((x1, y1), (x2, y2)), outline=tuple(color), width=2)
                draw.text((x1, y1 - 10), label, font=font, fill=tuple(color) + (255,))

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
