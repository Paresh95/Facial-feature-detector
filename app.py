import gradio as gr

def identity_function(input_image):
    return input_image

iface = gr.Interface(
    fn=identity_function,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Image(type="pil")
)

iface.launch()