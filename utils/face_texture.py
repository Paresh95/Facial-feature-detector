import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import dlib
import imutils
import os
from PIL import Image
from utils.cv_utils import get_image
from typing import Tuple


#TODO: face texture class - face detector and output face
#TODO: create YAML file to point towards static parameters
#TODO: Test main output and app
#TODO: Consider using other method for face detector - this one not as reliable
#TODO: Text output showing other examples - celeb, child, gender


class GetFaceTexture:
    def __init__(self) -> None:
        pass
    
    def preprocess_image(self, image) -> np.array:
        image = imutils.resize(image, width=800)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    
    def get_face(self, gray_image: np.array) -> np.array:
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray_image, 1)
        if len(faces) == 0:
            return "No face detected."

        x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
        face_image = gray_image[y:y+h, x:x+w]
        return face_image
    
    def get_face_texture(self, face_image: np.array) -> Tuple[np.array, float]:
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(face_image, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        variance = np.var(hist)
        std = np.sqrt(variance)
        return lbp, std
    
    def postprocess_image(self, lbp: np.array) -> Image:
        lbp = (lbp * 255).astype(np.uint8)
        return Image.fromarray(lbp)
    
    def main(self, image_input) -> Image:
        image = get_image(image_input)
        gray_image = self.preprocess_image(image)
        face_image = self.get_face(gray_image)
        lbp, std = self.get_face_texture(face_image)
        face_texture_image = self.postprocess_image(lbp)
        return face_texture_image, face_image, std


if __name__ == "__main__":    
    image_path = 'data/images_symmetry/gigi_hadid.webp'
    print(GetFaceTexture().main(image_path))
