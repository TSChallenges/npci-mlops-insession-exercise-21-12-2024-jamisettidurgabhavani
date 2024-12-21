import re
import joblib
import gradio
import gradio as gr
import numpy as np
import pandas as pd

# Load model
trained_model = joblib.load("app/model/iris_model.pkl")


# UI - Input componentsmodel/iris_model
in_sepal_l =gradio.Textbox(lines=1, placeholder=None, value="3.5", label='sepal length (cm)')
in_sepal_w =gradio.Textbox(lines=1, placeholder=None, value="4.0", label='sepal width (cm)')
in_petal_l =gradio.Textbox(lines=1, placeholder=None, value="2.5", label='petal length (cm)')
in_petal_w =gradio.Textbox(lines=1, placeholder=None, value="3.0", label='petal length (cm)')


# UI - Output component
out_label = gradio.Textbox(type="text", label="Species", elem_id="out_textbox")


# Label prediction function
def get_output_label(in_sepal_l, in_sepal_w, in_petal_l, in_petal_w):

    features = np.array([[in_sepal_l, in_sepal_w, in_petal_l, in_petal_w]])

    # Predict the class using the loaded model
    prediction = trained_model.predict(features)
    print(prediction)
    # Return result
    return ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]


# Create Gradio interface object
iface = gr.Interface(
    fn=get_output_label,  # Prediction function
    inputs=[in_sepal_l, in_sepal_w, in_petal_l, in_petal_w],  # Input components (sliders for flower features)
    outputs=[out_label],  # Output component (Textbox for the prediction)
    title="Iris Flower Classification",
    description="Iris Classification",
    flagging_mode='never'
)
# Launch gradio interface
iface.launch(server_name = "0.0.0.0", server_port = 7860)
                         # set server_name = "0.0.0.0" and server_port = 7860 while launching it inside container.
                         # default server_name = "127.0.0.1", and server_port = 7860
