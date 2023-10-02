import gradio as gr
import os
import yaml
from src.face_texture import GetFaceTexture
from src.face_symmetry import GetFaceSymmetry
from src.face_demographics import GetFaceDemographics
from src.face_proportions import GetFaceProportions


def combined_fn(input_image, input_image_2):
    demographics_dict = GetFaceDemographics().main(input_image)
    golden_ratios_dict, equal_ratios_dict, face_landmarks_image = GetFaceProportions().main(input_image)
    face_symmetry_image, symmetry_dict = GetFaceSymmetry().main(input_image)
    face_image, face_texture_image, texture_dict = GetFaceTexture().main(input_image)
    
    results = {
        "Demographic predictions": demographics_dict,
        "Face proportions (golden ratio)": golden_ratios_dict,
        "Face proportions (equal ratio)": equal_ratios_dict,
        "Face symmetry metrics": symmetry_dict,
        "Face texture metrics": texture_dict
    }
    with open("parameters.yml", 'r') as file:
        data = yaml.safe_load(file)
        results_interpretation = data["results_interpretation"]
    
    return (results, results_interpretation, face_image, face_landmarks_image, face_symmetry_image, face_texture_image)

gigi_hadid = os.path.join(os.path.dirname(__file__), "data/gigi_hadid.webp")
jay_z = os.path.join(os.path.dirname(__file__), "data/jay_z.jpg")

iface = gr.Interface(
    fn=combined_fn,
    inputs=[
        gr.Image(type="pil", label="Upload Face 1", value=jay_z),
        gr.Image(type="pil", label="Upload Face 2", value=gigi_hadid)
            ],
    outputs=[
        gr.JSON(label="Results"),
        gr.JSON(label="Results explainer"),
        gr.Image(type="pil", label="Extracted face"),
        gr.Image(type="pil", label="Face landmarks"),
        gr.Image(type="pil", label="Face symmetry"),
        gr.Image(type="pil", label="Extracted face texture"),         
    ],
    title="Advanced Facial Feature Detector",
    description=
    """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JSON Output in HTML</title>
        <style>
            .section {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
    
    <div class="section">
        <p><strong>Description:</strong> This tool analyses a facial image to predict age and gender, assess symmetry, evaluate proportions, and examine texture.</p>
        <p><strong>Instructions:</strong> For optimal results, upload a clear front-facing image (see example image). To do so, either drag and drop your photo or click on "Upload Face Image", then press 'Submit'.</p>
        <p><strong>Interpreting the results:</strong></p>
        <p><strong>Other information:</strong></p>
        <ul>
            <li>No uploaded photo is stored.</li>
            <li>The output will take several seconds to compute.</li>
            <li>If an error occurs try again or try a different photo or angle.</li>
        </ul>
    </div> 
    </body>
    </html>
    """,
    theme=gr.themes.Soft(),
    live=False,
)

iface.launch()
