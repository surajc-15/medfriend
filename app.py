from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import re
import json
from PIL import Image

app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def extract_and_save_data(text):
    # Regular expressions for extracting data
    patterns1 = {
        'Hemoglobin': r'Hemoglobin \(Hb\) (\d+\.\d+) Low',
        'RBC': r'Total RBC count - (\d+\.\d+)',
        'PCV': r'Packed Cell Volume \(PCV\) (\d+\.\d+)',  # Fixed typo
        'MCV': r'Mean Corpuscular Volume \(MCV\) (\d+\.\d+)',  # Fixed typo
        'MCH': r'MCH (\d+\.\d+)',  # Fixed typo
        'MCHC': r'MCHC (\d+\.\d+)',  # Fixed typo
        'WBC': r'Total WBC count - (\d+)',
        'Neutrophils': r'Neutrophils (\d+)',
        'Lymphocytes': r'Lymphocytes (\d+)',
        'Eosinophils': r'Eosinophils (\d+)',  # Fixed typo
        'Monocytes': r'Monocytes\. (\d+)',
        'Basophils': r'Basophils (\d+)',  # Fixed typo
        'Platelet': r'Platelet Count © #(\d+)'  # Fixed typo
    }

    patterns2 = {
        'Hemoglobin': r'Hemoglobin\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'RBC': r'Rec\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'HCT': r'HCT\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'MCV': r'MCV\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'MCH': r'MCH\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'MCHC': r'MCHC\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'WBC': r'WBC\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Neutrophils': r'NEU%\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Lymphocytes': r'LYM%\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Monocytes': r'MON%\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Eosinophils': r'EOS%\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Basophils': r'BAS%\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'Platelet': r'PLT\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?',
        'ESR': r'ESR\s*([\d.]+)\s*(?:\d+.\d+)?\s*(?:\w+)?'
    }

    # Initialize extracted data dictionary
    extracted_data = {}
    patterns = [patterns1, patterns2]

    for pattern_dict in patterns:
        for key, pattern in pattern_dict.items():
            match = re.search(pattern, text)
            if match:
                extracted_data[key] = match.group(1)
        if extracted_data:
            break

    return extracted_data

def medreport(pdf_path):
    # Convert PDF to image
    pages = convert_from_path(pdf_path)

    # Set the path to tesseract
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # Image preprocessing
    def preprocess_image(img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        processed_image = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv2.THRESH_BINARY, 61, 11)
        return processed_image

    # Preprocess the first page image
    img = preprocess_image(pages[0])

    # Extract text from image
    text = pytesseract.image_to_string(img, lang='eng')
    print(text)
    data = extract_and_save_data(text)
    print(data)
    return data

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join('uploads', pdf_file.filename)
    pdf_file.save(pdf_path)

    extracted_data = medreport(pdf_path)

    # Remove the uploaded file after processing
    os.remove(pdf_path)

    # Return extracted data as JSON
    return jsonify({"extracted_data": extracted_data})

if __name__ == '__main__':
    app.run(debug=True)
