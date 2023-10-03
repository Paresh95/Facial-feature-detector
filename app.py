import gradio as gr
import os
import yaml
import pandas as pd
from src.face_texture import GetFaceTexture
from src.face_symmetry import GetFaceSymmetry
from src.face_demographics import GetFaceDemographics
from src.face_proportions import GetFaceProportions
from PIL import Image as PILImage
from typing import List, Any


def get_results(image_input: PILImage.Image) -> List[Any]:
    demographics_dict = GetFaceDemographics().main(image_input)
    (
        ratios_dict,
        face_landmarks_image,
    ) = GetFaceProportions().main(image_input)
    face_symmetry_image, symmetry_dict = GetFaceSymmetry().main(image_input)
    face_image, face_texture_image, texture_dict = GetFaceTexture().main(image_input)

    results = {
        "Demographic predictions": demographics_dict,
        "Face proportions": ratios_dict,
        "Face symmetry metrics": symmetry_dict,
        "Face texture metrics": texture_dict,
    }

    return (
        results,
        face_image,
        face_landmarks_image,
        face_symmetry_image,
        face_texture_image,
    )


def concatenate_image(
    image_1: PILImage.Image, image_2: PILImage.Image
) -> PILImage.Image:
    image = PILImage.new("RGB", (image_1.width + image_2.width, image_1.height))
    image.paste(image_1, (0, 0))
    image.paste(image_2, (image_1.width, 0))
    return image


def get_dict_child_data(results_image: dict, image_number: int) -> dict:
    flattened_data = {"image": f"Face {image_number}"}
    for key, sub_dict in results_image.items():
        for sub_key, value in sub_dict.items():
            flattened_data[sub_key] = value
    return flattened_data


def output_fn(
    image_input_1: PILImage.Image, image_input_2: PILImage.Image
) -> List[Any]:
    with open("parameters.yml", "r") as file:
        data = yaml.safe_load(file)
        results_interpretation = data["results_interpretation"]

    if image_input_1 is not None and image_input_2 is not None:
        (
            results_image_1,
            face_image_1,
            face_landmarks_image_1,
            face_symmetry_image_1,
            face_texture_image_1,
        ) = get_results(image_input_1)
        (
            results_image_2,
            face_image_2,
            face_landmarks_image_2,
            face_symmetry_image_2,
            face_texture_image_2,
        ) = get_results(image_input_2)
        results_image_1, results_image_2 = get_dict_child_data(
            results_image_1, 1
        ), get_dict_child_data(results_image_2, 2)
        results_df = pd.DataFrame([results_image_1, results_image_2])
        face_image = concatenate_image(face_image_1, face_image_2)
        face_landmarks_image = concatenate_image(
            face_landmarks_image_1, face_landmarks_image_2
        )
        face_symmetry_image = concatenate_image(
            face_symmetry_image_1, face_symmetry_image_2
        )
        face_texture_image = concatenate_image(
            face_texture_image_1, face_texture_image_2
        )

    if image_input_1 == None and image_input_2 is not None:
        (
            results,
            face_image,
            face_landmarks_image,
            face_symmetry_image,
            face_texture_image,
        ) = get_results(image_input_2)
        results_df = pd.DataFrame([get_dict_child_data(results, 2)])

    if image_input_2 == None and image_input_1 is not None:
        (
            results,
            face_image,
            face_landmarks_image,
            face_symmetry_image,
            face_texture_image,
        ) = get_results(image_input_1)
        results_df = pd.DataFrame([get_dict_child_data(results, 1)])

    return (
        results_df,
        results_interpretation,
        face_image,
        face_landmarks_image,
        face_symmetry_image,
        face_texture_image,
    )


gigi_hadid = os.path.join(os.path.dirname(__file__), "data/gigi_hadid.webp")

iface = gr.Interface(
    fn=output_fn,
    inputs=[
        gr.Image(type="pil", label="Upload Face 1", value=gigi_hadid),
        gr.Image(type="pil", label="Upload Face 2"),
    ],
    outputs=[
        gr.DataFrame(label="Results"),
        gr.JSON(label="Results explainer"),
        gr.Image(type="pil", label="Extracted face"),
        gr.Image(type="pil", label="Face landmarks"),
        gr.Image(type="pil", label="Face symmetry"),
        gr.Image(type="pil", label="Extracted face texture"),
    ],
    title="Advanced Facial Feature Detector",
    description="""
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
        <font size="3">
        <h3><center>Turn your selfie into insights! Discover age and gender predictions, symmetry evaluations, and detailed proportions and texture analyses with our app.</center></h3>
        <hr style="margin-top: 20px; margin-bottom: 20px;">
        <p><strong>Instructions:</strong> Upload up to 2 photos. For optimal results, upload a clear front-facing image (see example). To do so, either drag and drop your photo or click <i>Upload Face</i>, then press <i>Submit</i>.</p>
        <p><strong>Other information:</strong></p>
        <ul>
            <li>The output computation requires approximately 5 to 30 seconds.</li>
            <li>No uploaded photo is stored.</li>
            <li>If an error occurs try again or try a different photo or angle.</li>
            <li>Once submitted, a section detailing the results and associated images will be displayed.</li>
        </ul>
        </font>  
    </div>
    </body>
    </html>
    """,
    theme=gr.themes.Soft(),
    live=False,
)

iface.launch()
