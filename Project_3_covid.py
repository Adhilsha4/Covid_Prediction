import streamlit as st
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import tempfile
import warnings
from PIL import Image
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Covid-19 App", layout="wide")

# Set background image from local file
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the background function
set_background("co2.jpg")

# --- Page 1: Home ---
def home_page():
    st.markdown(
        """
        <h1 style='text-align: center;
                   color: #FFFFFF;
                   font-weight: bold;
                   -webkit-text-stroke: 1px black;
                   text-shadow: 2px 2px 4px #000000;'>
            COVID-19 LUNG X-RAY CLASSIFICATION APP
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.image("img.png", width=400)

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>📌 Introduction</h2>
        <p style='text-align: justify; font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            The COVID-19 pandemic, which emerged in late 2019, has been one of the most disruptive global health crises 
            in recent history. It affected millions of lives—some were lost, and many continue to experience lasting 
            health impacts. Although the situation has improved globally, the virus still poses a threat with small 
            outbreaks in some areas.
        </p>
        <p style='text-align: justify; font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            COVID-19 primarily affects the lungs and can lead to viral pneumonia. Early detection is critical, especially 
            in resource-limited environments. To assist in early diagnosis, this deep learning-powered application 
            analyzes chest X-ray images and classifies them into three categories.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>🎯 Objective</h2>
        <ul style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            <li>To build a deep learning model that can detect COVID-19 using chest X-ray images.</li>
            <li>Classify images into three categories: <b>COVID Positive</b>, <b>COVID Negative</b>, and <b>Viral Pneumonia</b>.</li>
            <li>Deploy the model using a user-friendly web app built with Streamlit.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>🧰 Tools & Technologies</h2>
        <ul style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            <li><b>Deep Learning Framework:</b> Keras with TensorFlow backend</li>
            <li><b>Deployment:</b> Streamlit web app</li>
            <li><b>Dataset:</b> Kaggle COVID-19 Chest X-ray dataset</li>
            <li><b>Data Preprocessing:</b> Resized and cleaned 1,000 images (from 4,000 original images due to resource limits)</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>🧠 Model Architecture</h2>
        <p style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            The model is a Sequential Convolutional Neural Network (CNN) that extracts features and classifies the X-ray image. The architecture includes:
        </p>
        <ul style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            <li>Input Layer</li>
            <li>2 Convolutional Layers</li>
            <li>2 Pooling Layers</li>
            <li>Flatten Layer</li>
            <li>Fully Connected (Dense) Layer</li>
            <li>Output Layer with Softmax activation</li>
        </ul>
        <p style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            <b>Activation Functions:</b> ReLU in all hidden layers, Softmax in the output layer<br>
            <b>Performance:</b> Achieved ~90% accuracy with strong precision, recall, and F1-score
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>⚙️ How It Works</h2>
        <p style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            The user uploads a lung X-ray image via the app. The image is preprocessed and passed through the trained model. 
            The app then predicts whether the image belongs to a COVID-positive, COVID-negative, or viral pneumonia patient.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style='color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>🌍 Impact</h2>
        <p style='font-size: 16px; color: #FFFFFF; text-shadow: 1px 1px 2px #000000;'>
            This tool serves as an assistive diagnostic support system for healthcare professionals. It can help in 
            making quick preliminary assessments, especially in remote or under-resourced areas where expert radiologists 
            and lab tests may not be readily available.
        </p>
        """,
        unsafe_allow_html=True
    )


# --- Page 2: Input Features ---
def input_page():
    st.markdown(
        """
        <h1 style='text-align: center;
                   color: #FFFFFF;
                   font-weight: bold;
                   -webkit-text-stroke: 1px black;
                   text-shadow: 2px 2px 4px #000000;'>
            COVID PREDICTION USING LUNGS X-RAY IMAGE
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Load model
    model = load_model("my_model.keras", compile=False)

    # Upload image
    uploaded_file = st.file_uploader("Upload a lungs image", type=["jpg", "png", "jpeg"])

    # Prediction function
    def predict_img(img_array):
        img_resized = resize(img_array, (150, 150, 1))
        img_reshaped = img_resized.reshape(1, 150, 150, 1)
        pred = model.predict(img_reshaped)
        return pred.argmax()

    # Display result function
    def display_result(message, color):
        st.markdown(
            f"""
            <div style='
                background-color: rgba(255, 255, 255, 0.75); 
                color: {color}; 
                padding: 20px; 
                border-radius: 12px; 
                text-align: center; 
                font-size: 24px; 
                font-weight: bold;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                margin-top: 20px;
            '>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            img_array = imread(tmp_path)

        prediction = predict_img(img_array)
        if prediction == 0:
            display_result('⚠️ COVID Positive', 'red')
        elif prediction == 1:
            display_result('✅ COVID Negative', 'green')
        else:
            display_result('⚠️ COVID with Viral Pneumonia', 'orange')

# --- Page 3: Data Source ---
def Data_Source():
    st.markdown(
        """
        <h1 style='text-align: left;
                   color: #FFFFFF;
                   font-weight: bold;
                   -webkit-text-stroke: 1px black;
                   text-shadow: 2px 2px 4px #000000;'>
            DATA SOURCE
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write("Data Source:")
    st.link_button("Kaggle", "https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset")
    st.write('Colab Link:')
    st.link_button("Colab", "https://colab.research.google.com/drive/1SHdY7NLCMUPWc-EUCyFt0luzuGb7lGkg?usp=sharing")

# --- Navigation ---
pages = [
    st.Page(home_page, title="Home"),
    st.Page(input_page, title="Input Features"),
    st.Page(Data_Source, title="Data Source")
]

with st.sidebar:
    selected = st.navigation(pages, position="sidebar", expanded=True)

selected.run()

