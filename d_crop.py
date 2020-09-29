import time, cv2
from matplotlib import pyplot as plt
from detectors import TinyFace
from PIL import Image 
from utils import crop_thumbnail
import os

# load image with cv in RGB.
IMAGE_PATH = 'selfie.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = Image.open(IMAGE_PATH) 
# load detector.

DET = TinyFace(device='cpu')

# DSFD returns bboxes.
t = time.time()
bboxes = DET.detect_faces(img, conf_th=0.95)
print('detect %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
os.chdir('sample') 
# crop thumbnail from original image.
results = dict()
t = time.time()
for i, bbox in enumerate(bboxes):
    thumb_img, _ = crop_thumbnail(img, bbox, padding=1, size=100)
    results[str(i)] = thumb_img
    print(bbox[1])
    
    img_save = im.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
    img_save = img_save.save(str(i) + ".jpg")
print('crop %d faces in %.4f seconds.' % (len(results), time.time() - t))


