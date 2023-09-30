import cv2
import numpy as np


def get_image(image_input) -> np.array:
    """Outputs numpy array of image given a string filepath or PIL image"""
    if type(image_input) == str:
        image = cv2.imread(image_input)  # OpenCV uses BGR
    else:
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)  # PIL uses RGB
    return image
