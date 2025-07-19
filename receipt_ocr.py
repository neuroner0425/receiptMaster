from paddleocr import PaddleOCR
import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ExifTags
import math

lang = "korean"
filenames = os.listdir('./receipt_data/train/img')
print(filenames)

# ----------------------------------------------

def doOCR(lang, filename):
    file_name, file_extension = os.path.splitext(filename)
    img_path = os.path.join('receipt_data', 'train', 'img', filename)
    if not os.path.exists(img_path):
        assert "WTF"
    result_path = os.path.join('receipt_data', 'train', 'ocr_result', file_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ocr = PaddleOCR(lang=lang)
    result = ocr.predict(img_path)

    print(f"Result: {result}")

    for res in result:
        res.save_to_img(result_path)
        res.save_to_json(result_path)

for filename in filenames:
    doOCR(lang, filename)