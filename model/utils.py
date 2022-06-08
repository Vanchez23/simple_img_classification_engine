import requests

import cv2
import numpy as np

def load_img(img_link: str) -> np.ndarray:
    if img_link.startswith('http'):
        resp = requests.get(img_link, stream=True).raw
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, -1)
    else:
        img = cv2.imread(img_link)
    return img