from PIL import Image
import numpy as np
import sys
import os
import csv

num=3
find=['유리병','스티로폼','요쿠르트병','캔']
label=np.array([num])
myDir= "/Users/kjjs1/Documents/Tensorflow project/makedata/image/"+find[num]+"_resize/"
#Useful function
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList(myDir)

#print(myFileList)
for file in myFileList:
    print(file)
    img_file = Image.open(file)
    #img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')  # type: object
    #img_grey.save('result.png')
    #img_grey.show()

  # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    #print(value)
    value = value.flatten()
    value=np.append(value,label)
    print(value)
    with open("img_pixels.csv", 'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(value)