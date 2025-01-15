import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from collections import Counter
from fpdf import FPDF
import tempfile
import re
import os
import random
import pickle


with open('label_encoder.pkl', 'rb') as file:
    loaded_encoder = pickle.load(file)
print("Encoder loaded successfully!")
# Load the trained model
model = load_model(r'C:\LUMINAR\PROJECT\SmartGrocery\my_model2.keras')  # Replace with your model's path

# Update this path if necessary
path = r'C:\LUMINAR\PROJECT\SmartGrocery\archive (5)\validation'
categories = os.listdir(path)
# Generate a dictionary with random prices for each item
item_prices = {category: random.randint(5, 30) for category in categories}

# Function to preprocess images for model prediction


def preprocess_image(img):
      
      
      img2 = cv2.resize(img, (150, 150)).reshape(1, 150, 150, 3)/255.0
      return img2

# Function to sanitize text (remove emojis and non-ASCII characters)
def sanitize_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to generate bill content
def generate_bill_content(captured_items):
    item_counts = Counter(captured_items)
    total_price = sum(item_counts[item] * item_prices.get(item, 0) for item in item_counts)
    bill_lines = [["Item", "Quantity", "Price"]]
    for item, count in item_counts.items():
        price = count * item_prices.get(item, 0)
        bill_lines.append([sanitize_text(item.capitalize()), count, f"${price:.2f}"])
    return bill_lines, total_price

# Function to create a PDF for the bill
def create_pdf(bill_lines, total_price):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Smart Grocery Store", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Bill Receipt", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(70, 10, "Item", border=1, align="C")
    pdf.cell(40, 10, "Quantity", border=1, align="C")
    pdf.cell(50, 10, "Price", border=1, align="C")
    pdf.ln()
    pdf.set_font("Arial", size=12)
    for line in bill_lines[1:]:
        pdf.cell(70, 10, line[0], border=1)
        pdf.cell(40, 10, str(line[1]), border=1, align="C")
        pdf.cell(50, 10, line[2], border=1, align="R")
        pdf.ln()
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(110, 10, "Total", border=1, align="R")
    pdf.cell(50, 10, f"${total_price:.2f}", border=1, align="R")
    pdf.ln(20)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Make a payment here:", ln=True, align="C")
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 10, "https://smartgrocerystore/payment", ln=True, align="C", link="https://smartgrocerystore/payment")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf.output(temp_file.name)
        return temp_file.name

# Initialize session state for captured items
if 'captured_items' not in st.session_state:
    st.session_state.captured_items = []

# Streamlit UI
st.title("Smart Grocery Checkout System")
st.write("Capture items using your webcam or upload images to classify them.")

# Two-column layout
left_col, right_col = st.columns([1, 1])

# Left column: Webcam and image upload section
with left_col:
    st.header("📸 Capture or Upload")

    # Webcam capture
    capture_image = st.button("Capture Image")
    if capture_image:
        video = cv2.VideoCapture(0)
        time = 0
        st.text("Capturing...")
        while time <= 40:            
            success, image = video.read()
            if success:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_resise = cv2.resize(img_rgb, (150, 150))                
                img_reshape = img_resise.reshape(1, 150, 150, 3) / 255.0    
                prediction = model.predict(img_reshape)
                confidence = np.max(prediction)
                time+=1            
                if confidence>0.75:
                    video.release()
                    st.image(image, channels="BGR", caption="Captured Image")
                    p = prediction.argmax().item()
                    predicted_item = loaded_encoder.inverse_transform([p]).item()
                    # predicted_item = categories[prediction.argmax().item()]
                    print(predicted_item, prediction.argmax().item())
                    st.session_state.captured_items.append(predicted_item)
                    st.success(f"Item added: {predicted_item.capitalize()}")
                    break
            else:
                st.error("Unable to capture image. Please try again.")
        else:
            st.error("Unable to capture image. Please try again.")
        video.release()
    else:
        st.error("Unable to capture image. Please try again.")
        
    # Upload image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        try:
            image = np.array(image)
            image = cv2.resize(image, (150, 150))
            image = np.array(image).reshape(1, 150, 150, 3) / 255.0 
            prediction = model.predict(image)
            p = prediction.argmax().item()
            predicted_item = loaded_encoder.inverse_transform([p]).item()
            # predicted_item = categories[prediction.argmax()]
            st.write(f"Predicted Item: {predicted_item}")
            st.session_state.captured_items.append(predicted_item)
            st.success(f"Item added: {predicted_item.capitalize()}")
        except:
            st.error("Unable to capture image. Please try again.")

# Right column: Bill display and download section
with right_col:
    st.header("🧾 Bill")
    bill_lines, total_price = generate_bill_content(st.session_state.captured_items)

    if st.session_state.captured_items:
        for line in bill_lines:
            st.write(f"{line[0]:<15} {line[1]:<10} {line[2]:>10}")
        st.write(f"**Total: ${total_price:.2f}**")

        # Generate and download PDF
        pdf_path = create_pdf(bill_lines, total_price)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download Bill as PDF 🧾",
                data=pdf_file,
                file_name="SmartGrocery_Bill.pdf",
                mime="application/pdf"
            )
    else:
        st.write("No items captured yet.")

    # Clear captured items
    if st.button("Clear Items"):
        st.session_state.captured_items.clear()
        st.success("All items cleared.")
