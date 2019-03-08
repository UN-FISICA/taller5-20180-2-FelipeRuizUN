from cpython cimport array
from cython.view cimport array as cvarray
import numpy as np

def cof(double[:,:]x,i,j):#cofactores
    c=(x[(i+2)%3][(j+2)%3]*x[(i+1)%3][(j+1)%3]-x[(i+2)%3][(j+1)%3]*x[(i+1)%3][(j+2)%3])
    return(c)
def inv(double[:,:]x):#inversa de una matriz 
    cdef double det=0
    inver=cvarray(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(double), format="d")
    for k in range(x.shape[0]):
        det+=x[k][0]*cof(x,k,0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            inver[i][j]=1/det*cof(x,j,i)
    return(inver)
def mult(double[:,:]x,double [:,:]y):#multiplicación de matrices
    z=cvarray(shape=(x.shape[0],y.shape[1]), itemsize=sizeof(double), format="d")    
    for i in range (x.shape[0]):
        for j in range(y.shape[1]):
            z[i][j]=0
            for k in range(x.shape[1]):
                z[i][j]+=x[i][k]*y[k][j]
    return(z)
def arr(n,i):#crea un memory view de tamaño n, con todos los valores iguales a i 
    c=cvarray(shape=(n,1), itemsize=sizeof(double), format="d")    
    for k in range (n):
        c[k]=i
    cdef double[:,:] c_view=c
    return(c_view)
def prom( double[:,:] x):
    a = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]): 
            a+=x[i][j]
    return a/(x.shape[0]*x.shape[1])    
def bina(double[:,:] x):
    if prom(x)<255/2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]>35: x[i][j]=255
                else: x[i][j]=0
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]<200: x[i][j]=255
                else: x[i][j]=0   
    cdef double[:,:] x_view=x
    return(x_view)
def lbl(double[:,:] x):    
    cdef double l=1
        
    c_view=arr(x.shape[0],0)
    for i in range(x.shape[0]):                  
        for j in range(x.shape[1]):
            if x[i][j]>0:
                x[i][j]=l
                c_view[i][0]+=1
        if  c_view[i][0]==0 and c_view[i-1][0]!=0:              
            l+=1
    cdef double[:,:] x_view=x
    return(x_view,l)              

def cm(double[:,:] x,l):
    a=arr(int(l)-2,0)
    #cdef double [:] a_view=a    
    cdef float cn
    cdef float cm
    cdef float n
    for k in range(2,int(l)):
        n=0
        cm=0        
        for i in range(x.shape[0]):
            cn=0                           
            for j in range(x.shape[1]):
                if x[i][j]==k:
                    cn+=1
                    n+=1
            cm+=cn*i
        a[k-2]=cm/n
                     
    return(a)
def ace(double[:,:] x,hz,dx):    
    y=cvarray(shape=(x.shape[0],1), itemsize=sizeof(double), format="d")    
    t=cvarray(shape=(x.shape[0],1), itemsize=sizeof(double), format="d")               
    for i in range(x.shape[0]):
        y[i]=dx*x[i][0]
        t[i]=i/hz
    X=cvarray(shape=(3, x.shape[0]), itemsize=sizeof(double), format="d")
    cdef double[:,:] X_view=X
    for i in range(3):
        for j in range(x.shape[0]):
            X_view[i][j]=t[j][0]**i
            if i==2:
                X_view[i][j]=t[j][0]**i/2 
    Xt=cvarray(shape=(x.shape[0],3), itemsize=sizeof(double), format="d")
    cdef double[:,:] Xt_view=Xt
    for i in range(x.shape[0]):
        for j in range(3):
            Xt_view[i][j]=X_view[j][i]
    xtx=mult(X,Xt)
    xin=inv(xtx)
    b=mult(X,y)
    a=mult(xin,b)
    return(a[2][0])
def calc(image, hz, dx):
    img=lbl(bina(image))
    Y=cm(img[0],img[1])
    a=ace(Y,hz,dx)	
    return a
