def factorial(integer):
    """ Returns factorial of the given integer"""
    n = int(integer)
    if n<=1:
        return 1
    fact=1
    for i in range(1, n+1):
        fact*=i
    return fact

import gradio
gradio.Interface(factorial, inputs="text", outputs="text").launch(share=True)

# imported necessary libraries
import gradio as gr
import tensorflow as tf
import numpy as np
import requests
 
# loading the model
inception_net = tf.keras.applications.InceptionV3()
 
# Download human-readable labels.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")
 
def classify_image(image):
    """ Returns a dictionary with key as label and values
    as the predicted confidence for that label"""
    # reshaping the image
    image = image.reshape((-1, 299, 299, 3))
    # preprocessing the image for inception_v3
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    # predicting the output
    prediction = inception_net.predict(image).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}
 
# initializing the input component
image = gr.inputs.Image(shape = (299, 299, 3))
# initializing the output component
label = gr.outputs.Label(num_top_classes = 3)
 
# launching the interface
gr.Interface(fn = classify_image, inputs = image,
             outputs = label, capture_session = True).launch()

