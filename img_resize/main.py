import errno

from PIL import Image
import os, sys

find="우유곽"
path = "/Users/kjjs1/Documents/Tensorflow project/ImgDown/"+find+"_img/"
path2= "/Users/kjjs1/Documents/Tensorflow project/ImgDown/"+find+"_resize/"
try:
    os.makedirs(path2)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path2+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(f + ' resized.png', 'PNG', quality=100)
            print("DONE")

resize()