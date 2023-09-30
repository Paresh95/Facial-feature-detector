import dlib
import yaml
import cv2
import os
import numpy as np
import imutils
from src.cv_utils import get_image
from typing import List, Union
from PIL import Image as PILImage

with open("parameters.yml", "r") as stream:
    try:
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
        
class GetFaceProportions:
    def __init__(self):
        self.golden_ratio = 1.618
    
    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        image = imutils.resize(image, width=500)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    @staticmethod
    def detect_face_landmarks(gray_image: np.array) -> List[Union[np.array, np.array]]:
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(parameters["face_landmarks"]["model"])
        rects = detector(gray_image, 1)
        for rect in rects:
            shape = predictor(gray_image, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(gray_image, (x, y), 2, (0, 255, 0), -1)
            
        return shape, gray_image
    
    def compute_golden_ratios(self, shape: np.array) -> dict:
        top_mouth, middle_mouth, bottom_mouth = shape[51], shape[62], shape[57]
        top_nose, bottom_nose = shape[27], shape[33]
        bottom_chin = shape[8]
        
        # 1
        top_nose_to_middle_mouth_dist = np.linalg.norm(top_nose - middle_mouth) # euclidean distance
        middle_mouth_to_bottom_chin_dist = np.linalg.norm(middle_mouth - bottom_chin)
        ratio_top_nose_to_middle_mouth_vs_middle_mouth_to_bottom_chin = top_nose_to_middle_mouth_dist/middle_mouth_to_bottom_chin_dist
        
        # 2
        top_mouth_to_middle_mouth_dist = np.linalg.norm(top_mouth - middle_mouth)
        middle_mouth_to_bottom_mouth_dist = np.linalg.norm(middle_mouth - bottom_mouth)
        ratio_middle_mouth_to_bottom_mouth_vs_top_mouth_to_middle_mouth = middle_mouth_to_bottom_mouth_dist/top_mouth_to_middle_mouth_dist
    
        golden_ratios = {
            "Ideal ratio (golden ratio)": self.golden_ratio,
            "Top of nose to middle of mouth vs middle mouth to bottom of chin": ratio_top_nose_to_middle_mouth_vs_middle_mouth_to_bottom_chin,
            "Middle of mouth to bottom of mouth vs top of mouth to middle of mouth": ratio_middle_mouth_to_bottom_mouth_vs_top_mouth_to_middle_mouth
        }
        return golden_ratios
    
    @staticmethod
    def compute_equal_ratios(shape: np.array) -> dict:
        left_side_left_eye, right_side_left_eye, left_side_right_eye, right_side_right_eye = shape[36], shape[39], shape[42], shape[45]
        left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom = shape[37], shape[41], shape[44], shape[46]
        left_eyebrow_top, right_eyebrow_top = shape[19], shape[24]
        left_eye_center = np.mean([shape[37], shape[38], shape[41], shape[40]], axis=0)
        right_eye_center = np.mean([shape[43], shape[44], shape[47], shape[46]], axis=0)
        left_mouth, right_mouth = shape[48], shape[54]
        
        # 1
        left_eye_dist = np.linalg.norm(left_side_left_eye - right_side_left_eye)
        right_eye_dist = np.linalg.norm(left_side_right_eye - right_side_right_eye)
        average_eye_dist = (left_eye_dist + right_eye_dist)/2
        between_eye_dist = np.linalg.norm(right_side_left_eye - left_side_right_eye)
        ratio_eyes_width_vs_between_eye = average_eye_dist/between_eye_dist
        
        # 2
        left_eye_to_eyebrow_dist = np.linalg.norm(left_eyebrow_top - left_eye_top)
        right_eye_to_eyebrow_dist = np.linalg.norm(right_eyebrow_top - right_eye_top)
        eye_to_eyebrow_dist = (left_eye_to_eyebrow_dist + right_eye_to_eyebrow_dist)/2
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        eye_height = (left_eye_height + right_eye_height)/2
        ratio_eye_to_eyebrow_vs_eye_height = eye_to_eyebrow_dist/eye_height
        
        # 3
        left_to_right_eye_center_dist = np.linalg.norm(left_eye_center - right_eye_center)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        ratio_left_to_right_eye_center_vs_mouth_width = left_to_right_eye_center_dist/mouth_width
        
        equal_ratios = {
            "Ideal ratio": 1,
            "Eye width vs distance between eyes": ratio_eyes_width_vs_between_eye,
            "Eye to eyebrows vs eye height": ratio_eye_to_eyebrow_vs_eye_height,
            "Center of left to right eye vs mouth width": ratio_left_to_right_eye_center_vs_mouth_width
        }
        return equal_ratios
        

    def main(self, image_input):
        image = get_image(image_input)
        gray_image = self.preprocess_image(image)
        shape, image = self.detect_face_landmarks(gray_image)
        golden_ratios = self.compute_golden_ratios(shape)
        equal_ratios = self.compute_equal_ratios(shape)
        image = PILImage.fromarray(image)        
        return golden_ratios, equal_ratios, image
        

if __name__ == "__main__":
    path_to_images = "data/"
    image_files = os.listdir(path_to_images)
    for image in image_files:
        print(image)
        results = GetFaceProportions().main(path_to_images + image)
        print(results)