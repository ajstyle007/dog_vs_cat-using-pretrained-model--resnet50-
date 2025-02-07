import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import json

# Load the trained model

@st.cache_resource  # Cache the model for faster loading
def load_model():
    # model = torch.load("resnet50_model.pth") # we can do this also
    model = models.resnet50(pretrained=True)  # load the architecture
    model.fc = torch.nn.Linear(2048, 2)  # Adjust for 1000 ImageNet classes
    model.load_state_dict(torch.load("resnet50_weights.pth", map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

@st.cache_resource
def load_labels():
    with open("class_labels.json") as f:
        labels = json.load(f)
    return labels

class_labels = load_labels()

# Define image preprocessing function
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Streamlit UI
# st.title("🐱Image Classifier with ResNet-50")
st.markdown(
        "<h1 style='text-align: center; color: #D8125B; margin-top: -55px; font-weight: bold; font-size: 50px;'>Dog or Cat? Let's Find Out</h1>", 
        unsafe_allow_html=True
    )
st.markdown(
        "<h1 style='text-align: center; color: #FF921C; margin-top: -45px; font-weight: bold; font-size: 40px;'>with ResNet-50!</h1>", 
        unsafe_allow_html=True
    )
# st.write("Upload an image and let the model classify it!")
col1, col2 = st.columns([3,1])
col1.markdown(
        "<h1 style='text-align: left; color: #0FFCBE; margin-top: -20px; margin-bottom: -25px; font-weight: bold; font-size: 17px;'>Upload an image and let the model classify it!</h1>", 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<a href='https://example.com' target='_blank' style='text-decoration: none;'>"
        "<button style='background-color: #0FFCBE; color: black; border: none; padding: 8px 12px; font-size: 16px; cursor: pointer;'>Visit</button>"
        "</a>",
        unsafe_allow_html=True
    )


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    
        # Convert image to RGB (if it's RGBA or grayscale)
        if image.mode == "RGBA":
            image = image.convert("RGB")  # Remove alpha channel
        elif image.mode == "L":
            image = image.convert("RGB")  # Convert grayscale to RGB
        elif image.mode != "RGB":
            image = image.convert("RGB")


with col2:
    # Preprocess and predict
    if st.button("Predict"):
        image_tensor = process_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)   

        # Get top 2 predictions
        top_prob, top_class = torch.max(probabilities, dim=0)  # Get the highest probability

        # Get class name
        class_name = class_labels[top_class.item()]
        confidence = top_prob.item() * 100  # Convert to percentage

        # Display prediction
        # Get class name
        class_name = class_labels[top_class.item()]
        confidence = top_prob.item() * 100  # Convert to percentage
        if confidence < 98:  # If confidence is below 70%, classify as unknown
            st.warning("⚠️ The model is unsure. This image might not be a Dog or Cat.")
            st.write(f" **Confidence: {confidence:.2f}%**")

        else:
            st.success(f"🖼️ **Predicted Class: {class_name}**")
            st.success(f"🎯 **Confidence: {confidence:.2f}%**")