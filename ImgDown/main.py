import errno

from PIL import Image
import os, sys
from icrawler.builtin import GoogleImageCrawler

num=3
find=['유리병','스티로폼','요쿠르트병','캔']
google_crawler = GoogleImageCrawler(storage={'root_dir': find[num]+"_img"})
google_crawler.crawl(keyword=find[num], max_num=330)

path = "/Users/kjjs1/Documents/Tensorflow project/ImgDown/"+find[num]+"_img/"
path2= "/Users/kjjs1/Documents/Tensorflow project/ImgDown/"+find[num]+"_resize/"
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
            #print("DONE")

resize()
