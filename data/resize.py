
import cv2
import argparse
import os
import numpy
import sys
 
os.mkdir("resized")
for imgpath in sys.argv: 
  target_shape = (256, 256)
  print(imgpath)
  if imgpath.find('.jpg') == -1:
   continue

  img = cv2.imread(imgpath)
  height, width, depth = img.shape
  output_side_length=256
  new_height = output_side_length
  new_width = output_side_length
  if height > width:
    new_height = output_side_length * height / width
  else:
    new_width = output_side_length * width / height
  resized_img = cv2.resize(img, (new_width, new_height))
  height_offset = (new_height - output_side_length) / 2
  width_offset = (new_width - output_side_length) / 2
  cropped_img = resized_img[height_offset:height_offset + output_side_length,
  width_offset:width_offset + output_side_length]
  print("write")
  cv2.imwrite("resized/"+imgpath, cropped_img) 
