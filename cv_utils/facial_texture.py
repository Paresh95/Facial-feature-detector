import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import dlib
import imutils
import os
from PIL import Image


def compute_face_simplicity(image):
    
    ######## create if or depending on input - filepath or PIL file
    # Load the image from a filepath
    # image = cv2.imread(image_path)
    
    # Convert RGB to BGR format (OpenCV uses BGR while PIL uses RGB)
    image_np = np.array(image)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Resize the image
    image = imutils.resize(image, width=800)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/face_alignment/shape_predictor_68_face_landmarks.dat")
    
    # Detect the face in the image
    faces = detector(gray, 1)
    if len(faces) == 0:
        return "No face detected."

    x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    face_img = gray[y:y+h, x:x+w]
    
    
    # Parameters for LBP
    radius = 1
    n_points = 8 * radius
    
    # Apply LBP to the face region
    lbp = local_binary_pattern(face_img, n_points, radius, method="uniform")
    
    # Compute the histogram of the LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Measure the variance of the histogram
    variance = np.var(hist)
    std = np.sqrt(variance)
    print(std)
    
    # A hypothetical threshold - needs calibration
    threshold = 10000
    
    if std < threshold:
        simplicity = "Simple"
    else:
        simplicity = "Complex"
    
    # Visualizing the LBP pattern on the detected face
    # plt.imshow(lbp)
    lbp = (lbp * 255).astype(np.uint8)
    lbp = Image.fromarray(lbp)
    
    return lbp #, simplicity


if __name__ == "__main__":
    print(os.getcwd())
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/face_alignment/shape_predictor_68_face_landmarks.dat")
    print(predictor)
    
    image_path = 'data/images_symmetry/gigi_hadid.webp'
    print(compute_face_simplicity(image_path))
