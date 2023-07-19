import os

import cv2
import logging


logger = logging.getLogger("openvino_xai")
logger.setLevel(logging.INFO)

def save_explanations(dir_path, explanations):
    os.makedirs(dir_path, exist_ok=True)
    batch, classes, _, _ = explanations.shape
    for b in range(batch):
        for cl in range(classes):
            cv2.imwrite(os.path.join(dir_path, f"{b}_class_{cl}.jpg"), img=explanations[b, cl, :, :])
