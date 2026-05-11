import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# your imports
from model import ResNetTransformer
from explainability import GradCAM


# ===================== CONFIG =====================
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


# ===================== IMAGE TRANSFORM =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = ResNetTransformer(num_classes=4)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()

# Grad-CAM target layer (ResNet last conv block)
target_layer = model.backbone.backbone[-1]
cam_generator = GradCAM(model, target_layer)


# ===================== UI =====================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan to predict tumor type and visualize attention regions.")


uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:

    # ----------------------------
    # LOAD IMAGE
    # ----------------------------
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    input_tensor = transform(image).unsqueeze(0)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # ----------------------------
    # DISPLAY RESULT
    # ----------------------------
    st.subheader("🧾 Prediction")
    st.write(f"**Class:** {CLASS_NAMES[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # ----------------------------
    # GRAD-CAM
    # ----------------------------
    st.subheader("🔥 Model Attention (Grad-CAM)")

    cam = cam_generator.generate(input_tensor, class_idx=pred_class)
    cam_overlay = cam_generator.overlay(image_np, cam)

    st.image(cam_overlay, caption="Attention Heatmap", use_container_width=True)

    # ----------------------------
    # PROBABILITIES
    # ----------------------------
    st.subheader("📊 Class Probabilities")

    for i, p in enumerate(probs[0]):
        st.write(f"{CLASS_NAMES[i]}: {p.item():.3f}")
