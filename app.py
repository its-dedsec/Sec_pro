import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Attention
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2  # Import OpenCV
import os
from PIL import Image
import json
from datetime import datetime

# =============================================================================
# 1. Configuration and Global Variables
# =============================================================================

# Define image size and other constants
IMG_SIZE = 224
MODEL_PATH = "deepfake_detection_model.h5"  # Path to the saved model
ALERT_PATH = "security_alerts" # Path to save security alerts
os.makedirs(ALERT_PATH, exist_ok=True) # Ensure the directory exists

# Risk scores and descriptions (expand as needed)
RISK_SCORES = {
    "low": 3,
    "medium": 6,
    "high": 9,
    "critical": 10
}

RISK_DESCRIPTIONS = {
    "low": "Low risk. Deepfake likely used for non-malicious purposes (e.g., entertainment).",
    "medium": "Medium risk. Deepfake could be used for disinformation or minor fraud.",
    "high": "High risk. Deepfake likely used for significant fraud, reputational damage, or targeted attacks.",
    "critical": "Critical risk. Deepfake very likely used for highly damaging attacks, election interference, or major security breaches."
}

# Alert severities
ALERT_SEVERITIES = {
    "low": "Informational",
    "medium": "Warning",
    "high": "Alert",
    "critical": "Critical"
}

# =============================================================================
# 2. Model Definition (CNN with Attention)
# =============================================================================

def create_deepfake_detection_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Creates a CNN model for deepfake detection with an attention mechanism.

    Design Decisions:
    -   CNN architecture to automatically learn relevant features from images.
    -   Attention layer to focus on the most important parts of the image
        when making the classification.  This helps the model learn
        which facial regions are most likely to show deepfake artifacts.
    -   Relu activation for non-linearity.
    -   Dropout to prevent overfitting.
    -   Binary classification (real/fake) using sigmoid activation.
    """
    model = Sequential([
        # Convolutional layers to extract features
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        # Attention Layer
        Flatten(),
        Dense(128, activation='relu'),
        Attention(),  # Additive attention mechanism
        Dropout(0.5),
        # Output layer for binary classification (real/fake)
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================================================================
# 3. Model Loading
# =============================================================================

@st.cache_resource  # Use st.cache_resource for global resources like models
def load_deepfake_model():
    """Loads the pre-trained deepfake detection model.
       Handles the case where the model file might not exist.
    """
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file '{MODEL_PATH}' not found.  A default, untrained model will be used.  Please train a model and save it to this path for better results.")
        # Create a default model if the trained model is not found.
        model = create_deepfake_detection_model()
        return model  # Return the *untrained* model
    else:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}.  A default, untrained model will be used.")
            model = create_deepfake_detection_model()
            return model

deepfake_model = load_deepfake_model() # Load the model

# =============================================================================
# 4. Image Preprocessing
# =============================================================================

def preprocess_image(image):
    """
    Preprocesses the input image for the deepfake detection model.

    -   Resizes the image to the target size (IMG_SIZE x IMG_SIZE).
    -   Converts the image to an array.
    -   Normalizes the pixel values to the range [0, 1].
    -   Expands the dimensions to create a batch of size 1 (as the model expects).
    """
    try:
        if isinstance(image, str):  # Check if the image is a file path
            img = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
        elif isinstance(image, Image.Image): #check if it is PIL Image
            img = image.resize((IMG_SIZE, IMG_SIZE))
        else:
            img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img = Image.fromarray(img)

        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# =============================================================================
# 5. Deepfake Detection and Risk Assessment
# =============================================================================
def detect_deepfake_and_assess_risk(image):
    """
    Detects deepfakes in the input image and assesses the associated risk.

    -   Preprocesses the image.
    -   Uses the deepfake detection model to predict if the image is real or fake.
    -   Assigns a risk score and description based on the model's prediction.

    """
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return None, None, None, None # Return None values in case of error

    prediction = deepfake_model.predict(preprocessed_image)[0][0]  # Get the probability

    # Interpret the prediction (adjust thresholds as needed)
    if prediction < 0.1:
        label = "Real"
        risk_level = "low"
    elif prediction < 0.5:
        label = "Uncertain"
        risk_level = "medium"
    elif prediction < 0.8:
        label = "Fake"
        risk_level = "high"
    else:
        label = "Fake"
        risk_level = "critical"

    risk_score = RISK_SCORES[risk_level]
    risk_description = RISK_DESCRIPTIONS[risk_level]

    return label, prediction, risk_score, risk_description

# =============================================================================
# 6. Security Alert Generation
# =============================================================================

def generate_security_alert(image_path, label, prediction, risk_score, risk_description):
    """
    Generates a security alert in JSON format.

    -   Creates a dictionary containing the alert information.
    -   Dumps the dictionary to a JSON string.
    -   Saves the JSON string to a file.
    """
    timestamp = datetime.now().isoformat()
    severity = ALERT_SEVERITIES[list(RISK_SCORES.keys())[list(RISK_SCORES.values()).index(risk_score)]] # Get severity from risk score.

    alert_data = {
        "timestamp": timestamp,
        "image_path": image_path,
        "deepfake_detection": {
            "label": label,
            "confidence": float(prediction),  # Ensure float for JSON serialization
        },
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_level": list(RISK_SCORES.keys())[list(RISK_SCORES.values()).index(risk_score)],
            "risk_description": risk_description,
            "severity": severity
        },
        "alert_type": "Deepfake Detection",  # Add an alert type
        "status": "New", #Add a status
    }

    alert_filename = f"deepfake_alert_{timestamp.replace(':', '-')}.json"  # Use a safe filename
    alert_filepath = os.path.join(ALERT_PATH, alert_filename)
    try:
        with open(alert_filepath, "w") as f:
            json.dump(alert_data, f, indent=4)  # Pretty print JSON
        st.success(f"Security alert saved to: {alert_filepath}")
    except Exception as e:
        st.error(f"Error saving security alert: {e}")

# =============================================================================
# 7. Main Streamlit Application
# =============================================================================

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Deepfake Detection and Risk Assessment")

    # Check if OpenCV is installed
    if 'cv2' not in globals():
        st.error("OpenCV is not installed. Please install it using: `pip install opencv-python` and restart the app.")
        return  # Stop the app if OpenCV is not installed

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect deepfake and assess risk
        label, prediction, risk_score, risk_description = detect_deepfake_and_assess_risk(uploaded_file) # Pass the file object

        if label: # Check if label is not None (meaning detection was successful)
            # Display results
            st.subheader("Detection Results:")
            st.write(f"Label: {label}")
            st.write(f"Confidence: {prediction:.4f}")

            st.subheader("Risk Assessment:")
            st.write(f"Risk Score: {risk_score}")
            st.write(f"Risk Level: {list(RISK_SCORES.keys())[list(RISK_SCORES.values()).index(risk_score)]}")
            st.write(f"Risk Description: {risk_description}")

            # Generate and save security alert
            if label == "Fake" or label == "Uncertain":
                generate_security_alert(uploaded_file.name, label, prediction, risk_score, risk_description)
        else:
            st.error("Failed to process the image. Please try again.")

    st.subheader("About this App")
    st.markdown(
        """
        This Streamlit application detects deepfakes in images and assesses the associated cybersecurity risk.  It uses a Convolutional Neural Network (CNN) model to identify facial inconsistencies that are common in deepfake images.

        **Key Features:**
        -   Deepfake detection using a CNN model.
        -   Cybersecurity risk assessment based on the likelihood of the image being a deepfake.
        -   Generation of security alerts in JSON format.
        -   Simple and intuitive user interface.

        **How to Use:**
        1.  Upload an image using the file uploader.
        2.  The app will process the image and display the detection results and risk assessment.
        3.  If a deepfake is detected, a security alert will be generated and saved.

        **Important Notes:**
        -   The accuracy of the model depends on the quality and training data.
        -   This app is for research and demonstration purposes only.
        -   The risk assessment is based on heuristics and should not be considered definitive.
        -   **Important:** This app requires OpenCV. If you encounter an error, please install it using `pip install opencv-python` and restart the app.
        """
    )
    st.subheader("Hyperparameter Tuning Suggestions")
    st.markdown(
        """
        Here are some suggestions for hyperparameter tuning to improve the model's performance:
        -   **Learning Rate:** Experiment with different learning rates for the Adam optimizer (e.g., 0.0001, 0.001, 0.01).
        -   **Batch Size:** Adjust the batch size during training (e.g., 32, 64, 128).
        -   **Number of Epochs:** Train the model for more epochs to see if it improves accuracy.  Use early stopping to prevent overfitting.
        -   **Dropout Rate:** Try different dropout rates (e.g., 0.3, 0.5, 0.7) in the Dense layer.
        -   **Attention Layer Parameters**: If the Attention layer has parameters, experiment with those.
        -   **CNN Architecture:** Experiment with the number of convolutional layers, the number of filters in each layer, and the kernel size.
        -   **Regularization:** Add L1 or L2 regularization to the convolutional layers to prevent overfitting.
        -   **Data Augmentation:** Use more aggressive data augmentation techniques (e.g., more rotation, zoom, shear) during training.
        -   **Loss Function:** Explore other loss functions
        """
    )

    st.subheader("Potential Challenges and Limitations")
    st.markdown(
        """
        -   **Generalization:** The model may not generalize well to deepfakes created with different techniques or datasets.
        -   **Image Quality:** The model's performance can be affected by the quality of the input image (e.g., low resolution, noise).
        -   **Evolving Deepfake Techniques:** Deepfake technology is constantly evolving, so the model may need to be updated frequently to maintain accuracy.
        -   **Computational Resources:** Training deep learning models can require significant computational resources.
        -   **Explainability:** Deep learning models can be difficult to interpret, making it challenging to understand why a particular image is classified as a deepfake.
        -   **Adversarial Attacks:** Deepfake detection models can be vulnerable to adversarial attacks, where carefully crafted images can fool the model.
        """
    )
    st.subheader("Final Year Project Considerations")
    st.markdown(
        """
        For your final year project, here are some suggestions to make it even more impressive:
        -   **Improve Model Accuracy:** Focus on improving the accuracy of the deepfake detection model.  This could involve experimenting with different architectures, attention mechanisms, or training techniques.
        -   **Real-time Detection:** Explore implementing real-time deepfake detection using video streams.
        -   **Explainable AI (XAI):** Incorporate techniques to make the model's decisions more transparent and understandable.  For example, you could use visualization methods to highlight the facial regions that the model focuses on when making a prediction.
        -   **Multi-modal Detection:** Combine image analysis with other modalities, such as audio analysis, to improve detection accuracy.  Deepfakes often have audio inconsistencies as well.
        -   **Web API:** Create a web API that allows other applications to use your deepfake detection model.
        -   **User Interface:** Design a user-friendly interface for your application.
        -   **Ethical Implications:** Discuss the ethical implications of deepfake technology and the importance of deepfake detection.
        -   **Defense Strategies:** Research and implement defense strategies against deepfake attacks.
        -   **Dataset Curation:** Create a novel dataset of deepfakes
        -    **Comparative Analysis**: Compare the performance of different deepfake detection models or techniques.
        """
    )

if __name__ == "__main__":
    main()

