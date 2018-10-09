import json
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open('/home/yasin/python/chainer-gan-lib/SN-Full-PGGAN/result/base_cifar10_32'))

FID = []
IS = []
FID_smoothed = []
IS_smoothed = []
for i in range(len(data)):
    try:
        FID_smoothed.append(data[i]['FID_smoothed'])
        IS_smoothed.append(data[i]['IS_smoothed'])
        FID.append(data[i]['FID'])
        IS.append(data[i]['IS'])
    except:
        pass
    
FID_smoothed_array = np.asarray(FID_smoothed)
IS_smoothed_array = np.asarray(IS_smoothed)
FID_array = np.asarray(FID)
IS_array = np.asarray(IS)

x = range(len(FID_array))
plt.plot(x,FID_array,label='no_averaging')
plt.plot(x,FID_smoothed_array,label='averaging')
plt.legend(loc=1)
plt.xlabel('x10000 iteration')
plt.ylabel('FID')

x = range(len(IS_array))
plt.plot(x,IS_array,label='no_averaging')
plt.plot(x,IS_smoothed_array,label='averaging')
plt.legend(loc='bottom right')
plt.xlabel('x10000 iteration')
plt.ylabel('IS')


plt.plot(x,IS_array,x,IS_smoothed_array)

FID = sorted(FID)
IS = sorted(IS)
min5_FID = np.mean(np.asarray(FID[:5]))
max5_IS = np.mean(np.asarray(IS[5:]))

### ========

image_folder = '/home/yasin/python/chainer-gan-lib/SN-Full-PGGAN/result/base_cifar10_32_ws/'
images = [img for img in os.listdir(image_folder) if img.endswith("00.png")]
images.sort()
im_list = []
for i in range(len(images)):
    im_list.append(255*imread(os.path.join(image_folder,images[i])))

xs = np.asarray(im_list)

skvideo.io.vwrite("/home/yasin/base_cifar10_32_ws.mp4", xs)