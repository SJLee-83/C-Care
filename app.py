# app.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import cv2
import logging

# src 폴더의 DistancePredictor 클래스를 가져옵니다.
from src.predict import DistancePredictor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI 앱 생성
app = FastAPI(title="C-Care: Real-time Monitor Distance Predictor API")

# 모델 경로 설정
MODEL_PATH = "models/c-care_model.keras"

# 모델 로드 (앱이 시작될 때 한 번만 로드하여 메모리에 유지)
try:
    predictor = DistancePredictor(MODEL_PATH)
except Exception as e:
    logging.error(f"서버 시작 중 모델 로딩 실패: {e}")
    predictor = None

# 루트 경로 (서버 상태 확인용)
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>C-Care API</title>
        </head>
        <body>
            <h1>C-Care: VDT 증후군 예방 AI 모니터링 시스템 API</h1>
            <p>API 서버가 정상적으로 실행 중입니다. 예측을 원하시면 /docs 로 이동하여 테스트해주세요.</p>
        </body>
    </html>
    """

# 예측 엔드포인트
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 모니터와의 거리를 예측하고 결과를 반환합니다.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=500, detail="서버의 모델이 로드되지 않았습니다.")

    # 파일 확장자 확인
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="이미지 파일은 png, jpg, jpeg 형식만 지원합니다.")

    try:
        # 업로드된 파일 읽기
        image_data = await file.read()
        
        # 이미지 데이터를 numpy 배열로 변환
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # OpenCV로 이미지 디코딩 (BGR 형식)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
             raise HTTPException(status_code=400, detail="이미지 파일을 처리할 수 없습니다.")
        
        # OpenCV는 BGR, Keras 모델은 RGB를 사용하므로 색상 채널 순서 변경
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 예측 수행
        predicted_class, probability = predictor.predict(image_rgb)
        
        logging.info(f"예측 완료: class={predicted_class}, probability={probability:.2f}")

        # JSON 형태로 결과 반환
        return {
            "prediction": predicted_class,
            "probability": float(probability)
        }

    except Exception as e:
        logging.error(f"예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

# 이 스크립트가 직접 실행될 때 uvicorn 서버를 실행합니다.
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)