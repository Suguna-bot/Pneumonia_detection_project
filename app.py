import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sqlite3
import shutil
import tensorflow as tf
from tensorflow.keras import backend as K
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

app = Flask(__name__)

# Doctor details database
def init_doctor_db():
    connection = sqlite3.connect('doctors.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS doctors
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     specialization TEXT NOT NULL,
                     hospital TEXT NOT NULL,
                     phone TEXT NOT NULL,
                     email TEXT NOT NULL,
                     emergency_available BOOLEAN NOT NULL)''')
    
    # Insert sample doctors if table is empty
    cursor.execute("SELECT COUNT(*) FROM doctors")
    if cursor.fetchone()[0] == 0:
        sample_doctors = [
            ('Dr. Sarah Chen', 'Pulmonology', 'City General Hospital', '+1-555-0101', 's.chen@citygeneral.com', 1),
            ('Dr. Michael Rodriguez', 'Infectious Diseases', 'Metro Medical Center', '+1-555-0102', 'm.rodriguez@metromedical.com', 1),
            ('Dr. Emily Watson', 'Radiology', 'University Hospital', '+1-555-0103', 'e.watson@universityhospital.com', 1),
            ('Dr. James Wilson', 'Oncology', 'Cancer Care Center', '+1-555-0104', 'j.wilson@cancercare.com', 1),
            ('Dr. Lisa Thompson', 'Emergency Medicine', 'Emergency Care Hospital', '+1-555-0105', 'l.thompson@emergencycare.com', 1)
        ]
        cursor.executemany('''INSERT INTO doctors (name, specialization, hospital, phone, email, emergency_available)
                           VALUES (?, ?, ?, ?, ?, ?)''', sample_doctors)
    
    connection.commit()
    connection.close()

# GradCAM implementation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(image_path, heatmap, cam_path="static/gradcam.jpg", alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

# LIME implementation
def generate_lime_explanation(model, image_path, class_names):
    def model_predict(images):
        images = images.astype('float32') / 255.0
        return model.predict(images)
    
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype('double'), 
        model_predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    lime_path = "static/lime_explanation.jpg"
    plt.figure(figsize=(6, 6))
    plt.imshow(lime_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(lime_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return lime_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        name = request.form['name']
        password = request.form['password']
        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)
        result = cursor.fetchall()
        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')
    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)
        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)
        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()
        return render_template('index.html', msg='Successfully Registered')
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph', 'Confusion Matrix']
    return render_template('graph.html', images=images, content=content)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Clear previous images
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        
        fileName = request.form['filename']
        dst = "static/images"
        shutil.copy("test/"+fileName, dst)
        image_path = "static/images/"+fileName
        
        # Image processing
        img = cv2.imread("test/"+fileName)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        edges = cv2.Canny(img, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        retval2, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        # Load model and predict
        model = load_model('Lungdisease_ResNet.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            return img_array

        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            return predicted_class, confidence, input_image

        predicted_class, confidence, input_image = predict_single_image(image_path)
        
        # Generate GradCAM
        heatmap = make_gradcam_heatmap(input_image, model, 'conv2d_2')  # Adjust layer name as per your model
        gradcam_path = save_gradcam(image_path, heatmap)
        
        # Generate LIME explanation
        lime_path = generate_lime_explanation(model, image_path, class_names)
        
        # Get emergency doctors
        connection = sqlite3.connect('doctors.db')
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM doctors WHERE emergency_available = 1 LIMIT 3")
        doctors = cursor.fetchall()
        connection.close()
        
        # Disease descriptions
        disease_info = {
            'Cancer': {
                'description': 'Lung cancer is a type of cancer that begins in the lungs and can spread to other parts of the body.',
                'urgency': 'HIGH - Immediate consultation recommended'
            },
            'Covid19': {
                'description': 'COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus.',
                'urgency': 'HIGH - Immediate testing and isolation required'
            },
            'Pneumonia': {
                'description': 'Pneumonia is an infection that inflames the air sacs in one or both lungs.',
                'urgency': 'MEDIUM - Medical attention required within 24 hours'
            },
            'Tuberculosis': {
                'description': 'Tuberculosis is a serious infectious disease that mainly affects the lungs.',
                'urgency': 'HIGH - Immediate medical consultation needed'
            },
            'Normal': {
                'description': 'No significant abnormalities detected in the lung image.',
                'urgency': 'LOW - Routine follow-up recommended'
            }
        }
        
        info = disease_info.get(predicted_class, {'description': 'No information available', 'urgency': 'Consult doctor'})
        
        accuracy = f"The predicted image is {predicted_class} with a confidence of {confidence:.2%}"
        
        return render_template('results.html', 
                             status=predicted_class,
                             accuracy=accuracy,
                             description=info['description'],
                             urgency=info['urgency'],
                             ImageDisplay=f"http://127.0.0.1:5000/static/images/{fileName}",
                             ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                             ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                             ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                             GradCAM="http://127.0.0.1:5000/static/gradcam.jpg",
                             LIME="http://127.0.0.1:5000/static/lime_explanation.jpg",
                             doctors=doctors)
        
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    init_doctor_db()  # Initialize doctor database
    app.run(debug=True, use_reloader=False)
