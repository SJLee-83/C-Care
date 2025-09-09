# src/predict.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import logging

class DistancePredictor:
    def __init__(self, model_path):
        """모델을 로드하고 클래스 레이블을 초기화합니다."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.class_labels = ['under4', '4', 'over4']
            self.target_size = (224, 224)
            logging.info(f"모델 로딩 성공: {model_path}")
        except Exception as e:
            logging.error(f"모델 로딩 실패: {e}")
            self.model = None

    def preprocess_image(self, image_array):
        """OpenCV 이미지 배열을 모델 입력 형식에 맞게 전처리합니다."""
        # 모델이 (224, 224, 3) 크기를 기대하므로 리사이즈
        image = cv2.resize(image_array, self.target_size)
        image = image.astype("float32") / 255.0
        # 배치 차원 추가 (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_array):
        """전처리된 이미지에 대해 예측을 수행하고 결과를 반환합니다."""
        if self.model is None:
            return None, 0.0

        processed_image = self.preprocess_image(image_array)
        predictions = self.model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = self.class_labels[predicted_class_index]
        probability = np.max(predictions)
        
        return predicted_class_label, probability

# 이 스크립트가 직접 실행될 때 테스트를 위한 코드
if __name__ == '__main__':
    MODEL_PATH = "../models/c-care_model.keras"
    # TODO: 테스트할 이미지 경로로 수정하세요.
    IMAGE_PATH = "path/to/your/test_image.jpg" 

    predictor = DistancePredictor(MODEL_PATH)
    
    # OpenCV로 이미지 로드 (웹캠 프레임과 동일한 방식)
    test_image = cv2.imread(IMAGE_PATH)

    if test_image is not None and predictor.model is not None:
        predicted_class, probability = predictor.predict(test_image)
        print(f"Predicted Class: {predicted_class}")
        print(f"Probability: {probability:.2f}")
    else:
        print("테스트를 위한 이미지 또는 모델을 불러올 수 없습니다.")