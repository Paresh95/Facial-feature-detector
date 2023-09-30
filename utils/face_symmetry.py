import cv2
import numpy as np
from utils.cv_utils import get_image
from typing import Tuple, List, Union
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image as PILImage
import yaml

with open("parameters.yml", "r") as stream:
    try:
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class GetFaceSymmetry:
    def __init__(self):
        pass

    def get_faces(self, image: np.array) -> np.array:
        self.h, self.w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(300, 300))
        face_detector_net = cv2.dnn.readNetFromCaffe(
            parameters["face_detection"]["config"],
            parameters["face_detection"]["model"],
        )
        face_detector_net.setInput(blob)
        face_detections = face_detector_net.forward()
        return face_detections

    @staticmethod
    def postprocess_face(face: np.array) -> np.array:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)  # remove illumination
        face = cv2.GaussianBlur(face, (5, 5), 0)  # remove noise
        return face

    @staticmethod
    def get_face_halves(face: np.array) -> Tuple:
        mid = face.shape[1] // 2
        left_half = face[:, :mid]
        right_half = face[:, mid:]
        right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        right_half = cv2.flip(right_half, 1)
        return left_half, right_half

    @staticmethod
    def histogram_performance(
        left_half: np.array, right_half: np.array
    ) -> List[Union[float, float, float, float]]:
        hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_half], [0], None, [256], [0, 256])

        # Normalize histograms
        hist_left /= hist_left.sum()
        hist_right /= hist_right.sum()
        correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_INTERSECT)
        bhattacharyya = cv2.compareHist(
            hist_left, hist_right, cv2.HISTCMP_BHATTACHARYYA
        )

        return correlation, chi_square, intersection, bhattacharyya

    @staticmethod
    def orb_detector(left_half: np.array, right_half: np.array) -> int:
        """The fewer the matches (or the greater the average distance), the more dissimilar the images"""

        orb = cv2.ORB_create()
        keypoints_left, descriptors_left = orb.detectAndCompute(left_half, None)
        keypoints_right, descriptors_right = orb.detectAndCompute(right_half, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_left, descriptors_right)
        matches = sorted(matches, key=lambda x: x.distance)
        return len(matches)

    def get_face_similarity_results(
        self, left_half: np.array, right_half: np.array
    ) -> dict:
        structural_similarity, _ = ssim(left_half, right_half, full=True)
        cosine_distance = distance.cosine(left_half.ravel(), right_half.ravel())
        mse = mean_squared_error(left_half, right_half)
        mae = mean_absolute_error(left_half, right_half)
        (
            correlation,
            chi_square,
            intersection,
            bhattacharyya,
        ) = self.histogram_performance(left_half, right_half)
        matches = self.orb_detector(left_half, right_half)
        pixel_difference = np.sum((left_half - right_half) ** 2)

        d = {
            "structural_similarity": structural_similarity,
            "cosine_distance": cosine_distance,
            "mse": mse,
            "mae": mae,
            "histogram_correlation": correlation,
            "histogram_intersection": intersection,
            "orb_detector_matches": matches,
            "pixel_difference": pixel_difference,
        }
        return d

    def main(self, image_input) -> Tuple:
        image = get_image(image_input)
        face_detections = self.get_faces(image)
        lowest_mse = float("inf")
        best_face_data, best_left_half, best_right_half = None, None, None
        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence > 0.99:
                box = face_detections[0, 0, i, 3:7] * np.array(
                    [self.w, self.h, self.w, self.h]
                )
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                face = self.postprocess_face(face)
                left_half, right_half = self.get_face_halves(face)
                d = self.get_face_similarity_results(left_half, right_half)

            if d["mse"] < lowest_mse:
                best_face_data, best_left_half, best_right_half = (
                    d,
                    left_half,
                    right_half,
                )
                lowest_mse = d["mse"]

        full_face = np.hstack((best_left_half, best_right_half))
        full_face = PILImage.fromarray(full_face)

        return full_face, best_face_data


if __name__ == "__main__":
    image_path = "data/gigi_hadid.webp"
    results = GetFaceSymmetry().main(image_path)
    print(results)
