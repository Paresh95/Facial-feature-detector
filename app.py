import gradio as gr
from utils.face_texture import GetFaceTexture

iface = gr.Interface(
    fn=GetFaceTexture().main,
    inputs=gr.inputs.Image(type="pil"),
    outputs=[gr.outputs.Image(type="pil"),
             gr.outputs.Image(type="pil"),
             "text"
             ]
)

iface.launch()