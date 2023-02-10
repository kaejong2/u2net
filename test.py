import cv2
import glob
import os
import numpy as np

img_path = '/home/ljj/workspace/fit-ai-volume/detector/dataset/u2test/img/*'
res_path = '/home/ljj/U-2-Net/result/'
post = '/home/ljj/U-2-Net/post_result'

images = glob.glob(res_path+'*')

for i, img_path in enumerate(images[:1]):
    img = cv2.imread(img_path)
    res_mask = cv2.inRange(img, (180, 180, 180), (255,255,255))
    # bname = os.path.basename(img_path)
    # mask_path = os.path.join(res_path, bname)
    # mask = cv2.imread(mask_path)
    # mask = cv2.inRange(mask, 0.53, 1)

    cv2.imwrite(os.path.join(post, str(i)+ '.png'), res_mask)





