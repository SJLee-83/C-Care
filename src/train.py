# src/train.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import logging

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(directory, target_size=(224, 224), test_size=0.3, random_state=42):
    """데이터셋 폴더에서 이미지를 로드하고, 학습/테스트용으로 분할합니다."""
    images = []
    labels = []
    categories = {'under4': 0, '4': 1, 'over4': 2}

    logging.info("데이터 로딩을 시작합니다...")
    for category, label in categories.items():
        folder_path = os.path.join(directory, category)
        if not os.path.isdir(folder_path):
            logging.warning(f"폴더가 존재하지 않습니다: {folder_path}")
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = load_img(img_path, target_size=target_size)
            image = img_to_array(image)
            images.append(image)
            labels.append(label)

    images = np.array(images, dtype="float32") / 255.0  # 정규화
    labels = to_categorical(np.array(labels), num_classes=3)
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)
    logging.info(f"데이터 로딩 완료: Train={len(x_train)}개, Test={len(x_test)}개")
    
    return x_train, y_train, x_test, y_test

def build_model(num_classes):
    """EfficientNetB0 기반의 전이 학습 모델을 구축합니다."""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'), # elu 대신 relu도 좋은 선택입니다.
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_history(history):
    """학습 과정의 정확도와 손실을 시각화합니다."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """메인 학습 파이프라인 함수"""
    # 경로 설정
    DATA_DIR = "../data/Dataset/"
    MODEL_SAVE_PATH = "../models/c-care_model.keras" # 모델 저장 경로 및 이름 변경

    # 데이터 로드
    x_train, y_train, x_test, y_test = load_data(DATA_DIR)
    
    # 모델 빌드 및 컴파일
    model = build_model(num_classes=3)
    model.compile(optimizer=Nadam(learning_rate=0.001), # 학습률 조정
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # 콜백 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # 모델 학습
    logging.info("모델 학습을 시작합니다...")
    history = model.fit(x_train, y_train, 
                        batch_size=32, 
                        epochs=50, # Epochs 조정
                        validation_split=0.2,
                        callbacks=[early_stop, reduce_lr],
                        shuffle=True)
    
    # 모델 평가
    scores = model.evaluate(x_test, y_test, verbose=1)
    logging.info(f"Test Accuracy: {scores[1]*100:.2f}%")
    logging.info(f"Test Loss: {scores[0]:.4f}")

    # 결과 시각화
    plot_history(history)
    
    # 모델 저장
    model.save(MODEL_SAVE_PATH)
    logging.info(f"학습된 모델을 '{MODEL_SAVE_PATH}'에 저장했습니다.")


if __name__ == '__main__':
    main()