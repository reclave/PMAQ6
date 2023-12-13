import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Function to classify the image
def classify(image, model, class_names):
    # Preprocess the image (adjust as per your model's requirement)
    image = image.resize((224, 224))  # Example size, change as needed
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize if your model expects this
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    conf_score = np.max(predictions)

    # Get class name
    class_name = class_names[predicted_class[0]]

    return class_name, conf_score
    
# set title
st.title('Diagnosis Based on X-Ray Photos')

# set header
st.header('Upload an X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/keras_model.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### Confidence Score: {}%".format(int(conf_score * 1000) / 10))
