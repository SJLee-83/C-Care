# src/data_preprocessing.py

import os
import cv2
from retinaface import RetinaFace
import logging

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classify_and_move_faces(directory, target_size=(1280, 720)):
    """
    이미지 파일 이름의 첫 숫자를 기준으로 거리를 분류하고,
    얼굴이 인식되는 이미지만을 각 카테고리 폴더로 이동시킵니다.
    """
    categories = {'under4': [], '4': [], 'over4': []}
    
    # 각 카테고리에 대한 폴더 생성
    for category in categories.keys():
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            logging.info(f"'{category_path}' 폴더를 생성했습니다.")

    total_images = 0
    moved_images = 0
    
    # 원본 이미지가 있는 디렉토리만 순회
    # (하위 폴더는 제외하기 위해 listdir 결과를 미리 저장)
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for filename in filenames:
        if filename.lower().endswith(".jpg") and filename[0].isdigit():
            total_images += 1
            image_path = os.path.join(directory, filename)
            
            try:
                # 파일명에서 첫 번째 숫자 추출
                first_digit = int(filename[0])
                if first_digit < 4:
                    category = 'under4'
                elif first_digit == 4:
                    category = '4'
                else:
                    category = 'over4'

                # 얼굴 검출 (성공 시 dict, 실패 시 list 반환)
                faces = RetinaFace.detect_faces(image_path)
                if isinstance(faces, dict) and faces:
                    # 이미지 리사이즈 및 저장
                    image = cv2.imread(image_path)
                    if image is None:
                        logging.warning(f"이미지를 불러올 수 없습니다: {filename}")
                        continue
                    
                    image_resized = cv2.resize(image, target_size)
                    
                    dest_path = os.path.join(directory, category, filename)
                    cv2.imwrite(dest_path, image_resized)  # 리사이즈된 이미지 저장
                    os.remove(image_path)  # 원본 이미지 삭제
                    
                    moved_images += 1
                    logging.info(f"Moved {filename} to '{category}'")
                else:
                    logging.info(f"얼굴 미검출, {filename} 파일을 삭제합니다.")
                    os.remove(image_path) # 얼굴 없는 이미지는 삭제

            except Exception as e:
                logging.error(f"{filename} 처리 중 오류 발생: {e}")

    logging.info(f"총 {total_images}개의 이미지를 처리했습니다.")
    logging.info(f"얼굴이 검출되어 이동된 이미지는 {moved_images}개 입니다.")

# 이 스크립트가 직접 실행될 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # TODO: 자신의 데이터셋 경로로 수정하세요.
    DATASET_DIRECTORY = "../data/Dataset/" 
    classify_and_move_faces(DATASET_DIRECTORY)