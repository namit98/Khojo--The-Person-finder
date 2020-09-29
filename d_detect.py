import time, cv2
from matplotlib import pyplot as plt

from detectors import TinyFace

from utils import draw_bboxes



# load image with cv in RGB.
IMAGE_PATH = 'selfie.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load detectors.

DET = TinyFace(device='cpu')

# Tiny Face returns bboxes.
t = time.time()
bboxes = DET.detect_faces(img, conf_th=0.9, scales=[1])
print('Tiny Face : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img3 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))
import matplotlib.pyplot as plt



