# app/ocr.py
from pdf2image import convert_from_path
import pytesseract
import os

def pdf_to_text(pdf_path: str) -> str:
    texts = []
    for page in convert_from_path(pdf_path):
        texts.append(pytesseract.image_to_string(page))
    return "\n".join(texts)

def image_to_text(img_path: str) -> str:
    return pytesseract.image_to_string(img_path)
