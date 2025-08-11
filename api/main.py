import logging
import random
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.model_func import (
    class_img_to_label, load_img_model, transform_image, 
    load_txt_model, load_txt_tokenizer, text_preprocessing, class_text_to_label
)

logger = logging.getLogger('uvicorn.info')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Определение структур данных (модели Pydantic)

# Определение класса ответа для классификации изображений
class ImageResponse(BaseModel):
    class_name: str  # Название класса, например, dog, cat и т.д.
    class_index: int # Индекс класса из файла с индексами ImageNet

# Определение класса запроса для классификации текста
class TextInput(BaseModel):
    text: str  # Текст, введенный пользователем для классификации

# Определение класса ответа для классификации текста
class TextResponse(BaseModel):
    label: str  # Метка класса, например, positive или negative
    prob: float # Вероятность, связанная с меткой

# Определение класса запроса для табличных данных
class TableInput(BaseModel):
    feature1: float # Первая числовая характеристика
    feature2: float # Вторая числовая характеристика

# Определение класса ответа для табличных данных
class TableOutput(BaseModel):
    prediction: float # Предсказанное значение (например, 1 или 0)

# Глобальная переменная для модели
img_model = None  
txt_model = None
txt_tokenizer = None

# Загрузка моделей при запуске (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global img_model, txt_model, txt_tokenizer
    # Загрузка модели для изображений
    img_model = load_img_model().to(DEVICE)
    logger.info('Image model loaded')
    # Загрузка модели и токенизатора для текста
    txt_tokenizer = load_txt_tokenizer()
    txt_model = load_txt_model().to(DEVICE)
    logger.info('Text model loaded')
    # yield гарантирует корректную очистку ресурсов даже при возникновении ошибок в вашем API
    yield
    # Удаление моделей и освобождение ресурсов
    del img_model, txt_model, txt_tokenizer

app = FastAPI(lifespan=lifespan)

# Создание эндпоинтов (API Routes)
# Эндпоинты — это URL-адреса, 
# по которым можно обращаться к API для выполнения определённых действий
@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'Hello FastAPI!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    """
    Эндпоинт для классификации изображений.
    Принимает файл изображения, обрабатывает его, делает предсказание и возвращает название и индекс класса.
    """
    # Открытие изображения
    image = PIL.Image.open(file.file)
    # Предобработка изображения
    adapted_image = transform_image(image)
    # Логирование формы обработанного изображения
    logger.info(f'{adapted_image.shape}')
    # Предсказание класса изображения
    with torch.inference_mode():
        pred_index = img_model(adapted_image).numpy().argmax()
    # Преобразование индекса в название класса
    imagenet_class = class_img_to_label(pred_index)
    # Формирование ответа
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

# response_model для лучшей валидации
@app.post('/clf_text', response_model=TextResponse)
def clf_text(data: TextInput):
    """
    Эндпоинт для классификации текста.
    Случайно генерирует метку класса и вероятность для демонстрационных целей.
    """
    # Шаг 1: Предобработка текста
    processed_text = text_preprocessing(data.text)
    # Шаг 2: Используем ГЛОБАЛЬНЫЙ токенизатор
    inputs = txt_tokenizer(processed_text, return_tensors='pt', 
                           truncation=True, padding=True).to(DEVICE)
    # Шаг 3: Используем ГЛОБАЛЬНУЮ модель для предсказания
    with torch.no_grad():
        logits = txt_model(**inputs).logits
    proba = torch.sigmoid(logits).squeeze().item()
    # Шаг 4: Формирование ответа
    response = TextResponse(
        label=class_text_to_label(int(round(proba))),
        prob=proba
    )
    return response


# @app.post('/clf_table')
# def predict(x: TableInput):
#     """
#     Эндпоинт для классификации табличных данных.
#     Принимает значения признаков и возвращает предсказание модели.
#     """
    # # Преобразование признаков в массив и предсказание
    # prediction = sk_model.predict(np.array([x.feature1, x.feature2]).reshape(1, 2))
    # # Формирование ответа
    # result = TableOutput(prediction=prediction[0])
    # return result

if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)

##### 
# Проверка с помощью утилиты cURL:
# curl -X POST "http://127.0.0.1:8000/classify_image/" -L -H "Content-Type: multipart/form-data" -F "file=@dog.jpeg;type=image/jpeg"
#####
