import gradio as gr
from utils.face_texture import GetFaceTexture
from utils.face_symmetry import GetFaceSymmetry
from utils.face_demographics import GetFaceDemographics


def combined_fn(input_image):
    texture_results = GetFaceTexture().main(input_image)
    symmetry_results = GetFaceSymmetry().main(input_image)
    demographics_results = GetFaceDemographics().main(input_image)
    return (*texture_results, *symmetry_results, demographics_results)


iface = gr.Interface(
    fn=combined_fn,
    inputs=gr.inputs.Image(type="pil"),
    outputs=[
        gr.outputs.Image(type="pil"),  # From GetFaceTexture
        gr.outputs.Image(type="pil"),  # From GetFaceTexture
        "text",  # From GetFaceTexture
        gr.outputs.Image(type="pil"),  # From GetFaceSymmetry
        "text",  # From GetFaceSymmetry
        "text",  # From GetFaceDemographics
    ],
)

iface.launch()
