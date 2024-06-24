from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def case1():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Case 1</title>
            <meta charset="utf-8">
            
            <!-- bootstrap CDN link -->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>

        </head>
        <body>
            <div id="wrap" class="container">
                <!-- header -->
                <header class="d-flex">
                    <!-- 상단 로고 영역 -->
                    <div class="col-3 d-flex align-items-center">
                        <h1 class="font-weight-bold">
                            <a href="/" class="logo-text text-info">SafeZone</a>
                        </h1>
                    </div>
                    <!-- 상단 메뉴 영역 -->
                    <nav class="col-9 d-flex align-items-center">
                   
                    </nav>
                </header>

                <!-- section -->
                <section class="contents d-flex">
                    <section class="col-12">
                        <article class="h-50">
                            <h2 class="pt-5 ml-3">Case 1 페이지</h2>
                            <p>이 페이지에서는 Case 1 기능을 수행합니다.</p>
                        </article>
                    </section>
                </section>

                <!-- footer -->
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
                            고객센터:<br>
                            02-000-0000
                        </div>
                    </div>
                </footer>
            </div>
        </body>
    </html>
    """