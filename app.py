import gradio as gr

def identity_function(input_image):
    return input_image

iface = gr.Interface(
    fn=identity_function,
    inputs=gr.inputs.Image(),
    outputs=gr.outputs.Image()
)

iface.launch(share=True)