import cv2
import numpy as np
from cattlelogue.datasets import load_aglw_data, load_glw4_data

SHAPE = (540, 1080)

a = load_aglw_data(1961)[0]
a_shape = a.shape
a_v_dim = a.shape[0] * (SHAPE[1] / a_shape[1])
a = cv2.resize(a, (SHAPE[1], int(a_v_dim)), interpolation=cv2.INTER_AREA)
print(a.shape)
b = load_glw4_data(resolution=2)
print(b[0].shape)

# pad a downwards to match b's shape
if a.shape[0] < b[0].shape[0]:
    pad_height = b[0].shape[0] - a.shape[0]
    a = cv2.copyMakeBorder(a, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0)

#diff = cv2.merge((a, b[0]))
a = np.array(a>=-1000, dtype=np.uint8) * 255
cv2.imshow("a", a)
b = np.array(b[0]>=-1000, dtype=np.uint8) * 255
cv2.imshow("b", b)
cv2.imwrite("a.png", a)
cv2.imwrite("b.png", b)
#cv2.imshow("diff", diff)
cv2.waitKey(0)