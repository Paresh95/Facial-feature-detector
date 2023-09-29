import gradio as gr
from cv_utils.facial_texture import compute_face_simplicity

def identity_function(input_image):
    return input_image

iface = gr.Interface(
    fn=compute_face_simplicity,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Image(type="pil")
)

iface.launch()