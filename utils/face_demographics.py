import cv2
import yaml
import numpy as np
import os
from typing import Tuple
from utils.cv_utils import get_image


with open("parameters.yml", "r") as stream:
    try:
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class GetFaceDemographics:
    def __init__(self):
        pass
    
    @staticmethod
    def get_age(blob) -> Tuple:
        age_net = cv2.dnn.readNet(parameters["face_age"]["config"], parameters["face_age"]["model"])
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        age_net.setInput(blob)
        age_preds = age_net.forward()
        i = age_preds[0].argmax()
        age = age_list[i]
        age_confidence_score = age_preds[0][i]
        return age, age_confidence_score
    
    @staticmethod
    def get_gender(blob) -> Tuple:
        gender_net = cv2.dnn.readNet(parameters["face_gender"]["config"], parameters["face_gender"]["model"])
        gender_list = ['Male', 'Female']
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = gender_list[i]
        gender_confidence_score = gender_preds[0][i]
        return gender, gender_confidence_score

    def main(self, image_input) -> dict:
        image = get_image(image_input)      
        model_mean = (78.4263377603, 87.7689143744, 114.895847746) # taken from the model page on Caffe
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), model_mean, swapRB=False)
        age, age_confidence_score = self.get_age(blob)
        gender, gender_confidence_score = self.get_gender(blob)
        d = {
            "age_range": age,
            "age_confidence": age_confidence_score,
            "gender": gender,
            "gender_confidence": gender_confidence_score 
        }
        return d


if __name__ == "__main__":
    path_to_images = "data/"
    image_files = os.listdir(path_to_images)
    for image in image_files:
        print(image)
        results = GetFaceDemographics().main(path_to_images + image)
        print(results)