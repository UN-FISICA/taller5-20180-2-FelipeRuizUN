import numpy as np
import scipy.ndimage as nd
from scipy.misc import imread
from numpy.linalg import lstsq
def ace(image, hz, dx):
    imC2= 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    #Binarización de la blanco y negro
    if  np.average(imC2)<255/2:   #fondo negro
        CB=np.where(imC2<35,0,255)
    else:
        CB=np.where(imC2>200,0,255)
    imfil1=nd.median_filter(CB,(5,5)) #Filtro paso mediano
    kernel=1/25*np.ones((5,5),dtype=int)
    suave=nd.convolve(imfil1,kernel)
    lblim,n=nd.label(CB)
    ##Debo suavizarla para que em detecto solo las bolas sin los machones
    #kernel=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])   
    #bordes=nd.convolve(imC2,kernel)
    lbl,n = nd.label(suave)
    X,Y=[],[]
    for j in range(1,n+1):
        Y.append(nd.measurements.center_of_mass(suave, lbl, [j])[0][0])
     
    Y.sort()
    Y.pop(0)
    
    Y=np.array(Y)
    dt=np.ones(len(Y))
    for i in range(0,len(dt)):
        dt[i]=(i)/hz
    f=[]
    f.append(lambda x:np.ones_like(dt))
    f.append(lambda x:dt)
    f.append(lambda x:(1/2)*(dt**2))
    Xt=[]
    for fun in f:
        Xt.append(fun(dt))   
    Xt= np.array(Xt)
    X=Xt.transpose()    
    return(lstsq(X,Y,rcond=-1)[0][2])
