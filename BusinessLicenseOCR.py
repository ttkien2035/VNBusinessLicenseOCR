import cv2
import numpy as np
import editdistance
from paddleocr import PaddleOCR, draw_ocr
from time import time

t_start = time()
# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="vi", use_gpu=True) # The model file will be downloaded automatically when executed for the first time
img_path ='imgs_test/gpkd.jpg'
path=img_path.split('/')[-1]
label=path.split('.')[0]

result = ocr.ocr(img_path)
t_OCR=time()

txts = [line[1][0] for line in result]


list_editdistance=[]
for txt in txts:
    sub_txt=txt.split(":")
    list_editdistance.append(editdistance.eval(sub_txt[0],'Masodoanhnghiep'))
index=list_editdistance.index(min(list_editdistance))
position_text=txts[index]

resultMaso=txts[index].split(":")

print("Ma so doanh nghiep: ",resultMaso[1])
lines = ['Readme', 'How to write text files in Python']
with open('result/result_'+label+'.txt', 'w') as f:
    f.write("Ma so doanh nghiep:")
    f.write('\t')
    f.write(resultMaso[1])
    
t_editdistance=time()
# Visualization
flag_visualize=1

if flag_visualize:
    from PIL import Image
    image = Image.open(img_path).convert('RGB')
    result =list(filter(lambda result1: result1[1][0]==position_text,result))
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
   
    im_show = draw_ocr(image, boxes, txts, scores, font_path='fonts/arial.ttf')
    im_show = Image.fromarray(im_show)
    
    
    im_show.save('result/visualize_'+path)
    img = cv2. cvtColor(np.asarray(im_show),cv2. COLOR_RGB2BGR)
    cv2.imshow("visualize image",img)

t_total = time()
print('Timing: ------ OCR model ------ ', t_OCR - t_start)
print('Timing: ------ Edit distance ------ ', t_editdistance - t_OCR)
print('Timing: ------ Total cycle ------ ', t_total - t_start)
