import streamlit as st
import base64
from projects.stock_predictor_app import stock_predictor
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
import numpy as np

# Function to train and save the model
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('handwritten_digit.keras')

# Function to load the model and make predictions
def load_model_and_predict():
    loaded_model = load_model('handwritten_digit.keras')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    y_test = to_categorical(y_test, 10)

    predictions = loaded_model.predict(x_test)
    return predictions, y_test

# Streamlit app for ML Web App
def ml_web_app():
    st.title("Handwritten Digit Recognition")
    
    if st.button("Train and Save Model"):
        train_and_save_model()
        st.write("Model trained and saved successfully.")

    if st.button("Predict Digits"):
        predictions, y_test = load_model_and_predict()
        st.write("Predictions made successfully.")
        
        # Display the first prediction
        st.write(f"First prediction: {np.argmax(predictions[0])}, Actual label: {np.argmax(y_test[0])}")

        # Predict the label for the second test sample
        second_prediction = np.argmax(predictions[1])
        actual_second_label = np.argmax(y_test[1])
        st.write(f"Second prediction: {second_prediction}, Actual label: {actual_second_label}")

def home():
    # Page configs (tab title, favicon)
    st.set_page_config(
        page_title="Jason Byhring's Portfolio",
        page_icon="🍕",
        layout="wide",
    )

    # CSS styles file
    with open("styles/main.css") as f:
        st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Profile image file
    with open("./assets/Portfolio_Selfie.jpg", "rb") as img_file:
        img = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"

    # PDF CV file
    pdf_path = "C:/jbyhring_streamlit_python_portfolio/assets/Targeted Resume Version.pdf"
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # Read the logo image
    logo_path = "assets/supporting-member-badge.png"
    with open(logo_path, "rb") as img_file:
        logo_bytes = img_file.read()
        logo_base64 = base64.b64encode(logo_bytes).decode()

    # Sidebar with logo
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Navigation menu
    st.sidebar.title("Navigation")
    pages = {
        "Home": "home",
        "Projects": {
            "Stock Predictor App": "stock_predictor",
            "ML Web App": "ml_web_app"
        }
    }

    def render_navigation(pages, level=0):
        for name, key in pages.items():
            if isinstance(key, dict):
                st.sidebar.markdown(f"{'  ' * level}**{name}**")
                render_navigation(key, level + 1)
            else:
                if st.sidebar.button(name):
                    st.session_state.page = key

    if 'page' not in st.session_state:
        st.session_state.page = "home"

    render_navigation(pages)

    # Main content rendering based on the selected page
    if st.session_state.page == "home":
        st.markdown(f"""
            <div style="text-align: center;">
                <h1 style="font-size: 2.5em; font-weight: bold; margin-bottom: 0.5em;">Hi! My name is Jason Byhring 👋</h1>
                <img src="{img}" alt="Profile Image" style="border-radius: 50%; width: 200px;">
                <h2 style="font-weight: bold; margin-top: 0.5em;">Python Software Engineer</h2>
                <h3 style="font-weight: bold; margin-top: 0.5em;">AI Data Trainer</h3>
                <h3 style="font-weight: bold; margin-top: 0.5em;">Machine Learning Engineer</h3>
            </div>
        """, unsafe_allow_html=True)

        # Social Icons
        social_icons_data = {
            "LinkedIn": ["https://www.linkedin.com/in/jason-byhring-b233302a/", "https://cdn-icons-png.flaticon.com/512/174/174857.png"],
            "GitHub": ["https://github.com/byhringjstudent?tab=repositories", "https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg"],
            "Twitter": ["https://x.com/PyNuggets", "https://cdn-icons-png.flaticon.com/512/733/733579.png"],
        }

        social_icons_html = [
            f"<a href='{social_icons_data[platform][0]}' target='_blank' style='margin-right: 10px;'><img class='social-icon' src='{social_icons_data[platform][1]}' alt='{platform}' style='width: 30px; height: 30px;'></a>" 
            for platform in social_icons_data
        ]

        st.write(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            {''.join(social_icons_html)}
        </div>""", 
        unsafe_allow_html=True)

        st.write("##")

        # About me section
        st.subheader("About Me")
        st.write("""
        - 👨‍🎓 I am a **college student and Python Developer/AI Data Trainer** @ DataAnnotationCorp.
        - 💡 I am passionate about **Machine Learning/Artificial Intelligence, Data, Software Engineering.**
        - 📫 How to reach me: **byhringj@dupage.edu**
        """)

        st.write("##")

        # Download CV button
        st.download_button(
            label="📄 Download my CV",
            data=pdf_bytes,
            file_name="Targeted Resume Version.pdf",
            mime="application/pdf",
        )

        st.write("##")
        
        st.write("<div class=\"subtitle\" style=\"text-align: center;\">⬅️ Check out my Projects in the navigation menu! </div>", unsafe_allow_html=True)
    
    elif st.session_state.page == "stock_predictor":
        st.title("Stock Predictor App")
        st.write("""
        The Stock Predictor App is a web application that allows users to predict stock prices using machine learning algorithms. It provides insights and forecasts based on historical stock data.
        """)
        stock_predictor()  # Call the function directly
    
    elif st.session_state.page == "ml_web_app":
        ml_web_app()

if __name__ == '__main__':
    home()
