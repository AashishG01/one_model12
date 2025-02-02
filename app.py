import streamlit as st
import pytesseract
import cv2
from paddleocr import PaddleOCR
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page title
st.title("Medical Report & Disease Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_path = "temp.jpg"
    image.save(image_path)

    # Extract text using pytesseract
    def extract_text(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 3)
        text = pytesseract.image_to_string(img)
        return text

    # Extract tables using PaddleOCR
    def extract_table(image_path):
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        result = ocr.ocr(image_path, cls=True)
        extracted_text = "\n".join([line[1][0] for line in result[0]])
        return extracted_text

    # Display extracted text
    if st.button("Extract Text"):
        extracted_text = extract_text(image_path)
        st.text_area("Extracted Text", extracted_text, height=200)

    if st.button("Extract Table"):
        extracted_table = extract_table(image_path)
        st.text_area("Extracted Table", extracted_table, height=200)

    # Chest X-ray classification
    def classify_xray(image_path):
        model = timm.create_model("densenet121", pretrained=True, num_classes=14)
        model.eval()
        disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
        ]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        probabilities = torch.sigmoid(output).squeeze().numpy()
        results = {disease: round(prob, 2) for disease, prob in zip(disease_labels, probabilities) if prob > 0.5}
        return results if results else "No significant disease detected."

    if st.button("Classify Chest X-ray"):
        xray_result = classify_xray(image_path)
        st.write(xray_result)

    # Brain tumor detection
    def detect_brain_tumor(image_path):
        model_name = "ShimaGh/Brain-Tumor-Detection"
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        confidence = torch.softmax(logits, dim=1).max().item()
        return {"prediction": label, "confidence": confidence}

    if st.button("Detect Brain Tumor"):
        brain_result = detect_brain_tumor(image_path)
        st.write(brain_result)

    # Skin disease classification
    def classify_skin_disease(image_path):
        model_name = "WahajRaza/finetuned-dermnet"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
        disease_name = model.config.id2label.get(predicted_label, "Unknown Disease")
        return {"predicted_disease": disease_name}

    if st.button("Classify Skin Disease"):
        skin_result = classify_skin_disease(image_path)
        st.write(skin_result)
