from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from routers import case1, case2

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(case1.router, prefix="/case1", tags=["case1"])
app.include_router(case2.router, prefix="/case2", tags=["case2"])

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>SafeZone</title>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
        </head>
        <body>
            <div id="wrap" class="container">
                <header class="d-flex mb-4 mt-4">
                    <div class="col-5"></div>
                    <div class=" d-flex align-items-center">
                        <h1 class="font-weight-bold">
                            <a href="/" class="logo-text text-info">SafeZone</a>
                        </h1>
                    </div>
                    <div class=""></div>
                </header>
                <section class="contents d-flex">
                    <section class="col-6">
                        <img src="https://cdn.pixabay.com/photo/2017/10/26/17/51/under-construction-2891888_1280.jpg" alt="메인 로고" width="530" height="480">
                    </section>
                    <section class="col-6">
                        <article class="h-50">
                            <h2 class="pt-5 ml-3">안전을 책임지는 세이프존</h2>
                            <h2 class="pt-5 ml-3">2가지 기능을 제공합니다</h2>
                        </article>
                        <article class="h-50 d-flex justify-content-around align-items-center">
                            <button type="button" class="btn btn-info col-6 mr-2" onclick="location.href='/case1'">case1</button>
                            <button type="button" class="btn btn-info col-6 ml-2" onclick="location.href='/case2'">case2</button>
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
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
