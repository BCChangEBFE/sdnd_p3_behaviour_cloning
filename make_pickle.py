import pickle
import matplotlib.image as mpimg
from PIL import Image
import scipy.misc as misc
import time

x = []
y = []

f_in = open('driving_log.csv','r')
line = f_in.readline()


while True:
    line = f_in.readline()
    print(line.split(","))
    if not line:
        break
    center,left,right,steering,throttle,brake,speed = line.split(",")
    y.append( float(steering) )
    #x.append( misc.imresize(mpimg.imread(center),(32,64)) )
    #x.append( misc.imresize(mpimg.imread(center),(64,128)) )
    #x.append( misc.imresize(mpimg.imread(center),(160,320)) )
    x.append( mpimg.imread(center) )


data = {'features':x, 'results':y}
#fout = open('data.p','bw+')
#pickle.dump(data,fout, protocol=4)
#pickle.dump(data,fout)

pickle.dump( data, open( "data.p", "wb" ) )   
#pickle.dump( data, open( "data.p", "wb" ), protocol=2 )   

