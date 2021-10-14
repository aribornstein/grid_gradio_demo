import gradio as gr
import numpy as np
from flash.image import ImageClassifier

# 1. Load Model
model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")
image = gr.inputs.Image(shape=(299, 299))
label = gr.outputs.Label(num_top_classes=1)

def classify(img):
    img = np.transpose(img, (2, 0, 1))
    return model.predict(img, data_source="numpy")[0]

gr.Interface(fn=classify, inputs=image, outputs=label, capture_session=True).launch()