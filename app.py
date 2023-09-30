import gradio as gr
import os
from src.face_texture import GetFaceTexture
from src.face_symmetry import GetFaceSymmetry
from src.face_demographics import GetFaceDemographics
from src.face_proportions import GetFaceProportions


def combined_fn(input_image):
    texture_results = GetFaceTexture().main(input_image)
    symmetry_results = GetFaceSymmetry().main(input_image)
    demographics_results = GetFaceDemographics().main(input_image)
    proportion_results = GetFaceProportions().main(input_image)
    return (*texture_results, *symmetry_results, demographics_results, *proportion_results)

gigi_hadid = os.path.join(os.path.dirname(__file__), "data/gigi_hadid.webp")

iface = gr.Interface(
    fn=combined_fn,
    inputs=gr.Image(type="pil", label="Upload Face Image", value=gigi_hadid),
    outputs=[
        gr.Image(type="pil", label="Extracted face"),
        gr.Image(type="pil", label="Extracted face texture"), 
        "json",
        gr.Image(type="pil", label="Face symmetry"),  
        "json",
        "json",
        "json",
        "json",
        gr.Image(type="pil", label="Face landmarks"),
    ],
    title="Advanced Facial Feature Detector",
    description="A comprehensive tool for detailed face analysis. Please upload a clear face image for best results.",
    theme=gr.themes.Soft(),
    live=False,
)

iface.launch()
