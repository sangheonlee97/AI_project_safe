from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
import tempfile
import base64
from PIL import Image, ImageDraw
from ultralytics import YOLO
import openai
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

router = APIRouter()

class_names = [
    '소화기', '화재위험', '보행금지', '미지정위험', '생물학적위험', '고양이', '건설차량', '부식위험', '개', '음료금지', 
    '귀덮개', '감전위험', '폭발위험', '응급처치', '지게차', '지게차위험', '안경', '장갑', '경비견', '하네스', '하이바', 
    '레이저빔', '마스크', '불꽃', '굴착금지', '귀덮개미착용', '안경미착용', '장갑미착용', '하이바미착용', '마스크미착용', 
    '신발미착용', '흡연금지', '조끼미착용', '비전리방사선', '문열림위험', '작업자', '전화사용', '방사선위험', '신발', 
    '독성위험', '안전조끼', '안경착용위험', '호흡기착용'
]

class_dict = {name: idx for idx, name in enumerate(class_names)}

class YOLODetectionTool:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        np.random.seed(0)
        self.colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
        self.class_id = None

    def set_class_name(self, class_name):
        if class_name in class_dict:
            self.class_id = class_dict[class_name]
        else:
            raise ValueError(f"클래스 이름 '{class_name}'에 해당하는 ID를 찾을 수 없습니다.")

    def run(self, frame):
        if self.class_id is None:
            raise ValueError("탐지할 클래스가 설정되지 않았습니다.")
        
        results = self.model(frame)
        predictions = results[0].boxes

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        for pred in predictions:
            cls = int(pred.cls[0])
            if cls == self.class_id:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                conf = pred.conf[0]
                label = f"{class_names[cls]} {conf:.2f}"
                color = self.colors[cls].tolist()
                draw.rectangle(((x1, y1), (x2, y2)), outline=tuple(color), width=2)
                draw.text((x1, y1 - 10), label, fill=tuple(color) + (255,))

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# OpenAI API 키 설정 (환경 변수나 직접 입력)

# 프롬프트 템플릿 정의
template = """
    너는 욜로 모델의 라벨을 맞추는 모델이야. 문장을 입력받으면 아래의 라벨들 중에서 그 문장이 설명하는 라벨을 한개만 정해줘. 라벨이기 때문에 너가 말하는 결과에 따옴표나 컴마(,), 점(.)이 포함되면 안돼.
    소화기, 화재위험, 보행금지, 미지정위험, 생물학적위험, 고양이, 건설차량, 부식위험, 개, 음료금지, 
    귀덮개, 감전위험, 폭발위험, 응급처치, 지게차, 지게차위험, 안경, 장갑, 경비견, 하네스, 하이바, 
    레이저빔, 마스크, 불꽃, 굴착금지, 귀덮개미착용, 안경미착용, 장갑미착용, 하이바미착용, 마스크미착용, 
    신발미착용, 흡연금지, 조끼미착용, 비전리방사선, 문열림위험, 작업자, 전화사용, 방사선위험, 신발, 
    독성위험, 안전조끼, 안경착용위험, 호흡기착용
    
    description:{description}"""
prompt = PromptTemplate(template=template, input_variables=["description"])

# LLM 구성 (여기서는 OpenAI GPT-3.5-turbo 모델을 사용)
llm = OpenAI(model_name="gpt-3.5-turbo")

# YOLO Tool 구성
yolo_tool = YOLODetectionTool(model_path="best.pt")

class CustomLLMChain:
    def __init__(self, llm, prompt, yolo_tool):
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.yolo_tool = yolo_tool
        self.current_description = ""
        self.current_class_name = ""

    def __call__(self, inputs):
        description = inputs['description']
        frame = inputs['frame']

        # 입력된 설명이 변경되었을 때만 LLM을 호출하여 클래스 이름 추출
        if description != self.current_description:
            response = self.llm_chain.run({"description": description})
            class_name = response.strip()
            self.current_class_name = class_name
            self.current_description = description

            # YOLO Tool에 클래스 이름 설정
            self.yolo_tool.set_class_name(self.current_class_name)

        # YOLO Tool을 사용하여 객체 탐지 및 주석 추가
        annotated_frame = self.yolo_tool.run(frame)
        return {"annotated_frame": annotated_frame}

# CustomLLMChain 구성
chain = CustomLLMChain(llm=llm, prompt=prompt, yolo_tool=yolo_tool)

@router.get("/", response_class=HTMLResponse)
async def case2():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Case 2</title>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
        </head>
        <body>
            <div id="wrap" class="container">
                <header class="d-flex">
                    <div class="col-3 d-flex align-items-center">
                        <h1 class="font-weight-bold">
                            <a href="/" class="logo-text text-info">SafeZone</a>
                        </h1>
                    </div>
                    <nav class="col-9 d-flex align-items-center"></nav>
                </header>
                <section class="contents d-flex">
                    <section class="col-12">
                        <article class="h-50">
                            <h2 class="pt-5 ml-3">Case 2 페이지</h2>
                            <form id="upload-form" class="mt-3" enctype="multipart/form-data">
                                <div class="form-group">
                                    <label for="description">Enter the object description (e.g., '조끼 안 입은 사람 찾아줘')</label>
                                    <input type="text" class="form-control" id="description" name="description" required>
                                </div>
                                <div class="form-group">
                                    <label for="file">Upload a video file</label>
                                    <input type="file" class="form-control-file" id="file" name="file" accept="video/*" required>
                                </div>
                                <button type="submit" class="btn btn-primary">Process Video</button>
                            </form>
                            <div class="mt-5" id="video-result">
                                <!-- 비디오 결과가 여기에 표시됩니다 -->
                            </div>
                        </article>
                    </section>
                </section>
                <footer class="d-flex">
                    <div class="col-2 d-flex justify-content-center align-items-center">
                        <div class="footer-logo text-dark font-weight-bold">SafeZone</div>
                    </div>
                    <address class="col-8 text-center">
                        <div class="mt-4">
                            Copyright © safezone 2024
                        </div>
                    </address>
                    <div class="col-2 d-flex justify-content-center align-items-center">
                        <div class="font-weight-bold">
                            고객센터:<br>02-000-0000
                        </div>
                    </div>
                </footer>
            </div>
            <script>
                document.getElementById('upload-form').addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    const response = await fetch('/case2/process-video', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    const videoResultDiv = document.getElementById('video-result');
                    videoResultDiv.innerHTML = '';

                    if (data && data.encoded_video) {
                        const videoElement = document.createElement('video');
                        videoElement.controls = true;
                        videoElement.src = `data:video/mp4;base64,${data.encoded_video}`;
                        videoElement.style.width = '100%';
                        videoResultDiv.appendChild(videoElement);
                    } else {
                        videoResultDiv.innerHTML = '<p>비디오 처리 중 오류가 발생했습니다.</p>';
                    }
                });
            </script>
        </body>
    </html>
    """

@router.post("/process-video")
async def process_video(description: str = Form(...), file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # LangChain을 통해 YOLO Tool을 호출하여 클래스 이름 설정
    chain({"description": description.strip(), "frame": np.zeros((height, width, 3), dtype=np.uint8)})

    output_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = chain({"description": description.strip(), "frame": frame})
        annotated_frame = result["annotated_frame"]
        output_frames.append(annotated_frame)

    cap.release()

    # 결과 비디오 저장
    output_video_path = temp_file_path.replace(".mp4", "_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(frame)

    out.release()

    # 비디오 파일을 base64로 인코딩
    with open(output_video_path, "rb") as video_file:
        encoded_video = base64.b64encode(video_file.read()).decode('utf-8')

    return JSONResponse({"encoded_video": encoded_video})
