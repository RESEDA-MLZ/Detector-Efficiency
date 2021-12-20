import numpy as np
from scipy import integrate

import matplotlib.pyplot as py

# functions from "Efficiency formulas.txt"

na=6.022*10**(23)
rho=2.34
a=10.81
sigma=3837.

def mu(lam, iso):
    return rho * na/a * sigma*iso/100*10**(-24) / 10000*lam/1.8
	
def Pvor(x, d, r, alpha):
    return (1 - (d - x*np.cos(alpha))/r)/2

def Prück(x, r, alpha):
    return (1 - x*np.cos(alpha)/r)/2

def integer1(x, lam, iso, d, r, alpha):
    return mu(lam, iso)*np.exp(-mu(lam,  iso)*x)*Pvor(x, d, r, alpha)

def integer2(x, lam, iso, r, alpha):
    return mu(lam, iso)*np.exp(-mu(lam,  iso)*x)*Prück(x, r, alpha)

def B10SchraegVor(d, u, o, r, lam, alpha, iso):
    return integrate.quad(integer1, u, o, args=(lam, iso, d, r, alpha))[0]
		
def B10SchraegRück(u, o, r, lam, alpha, iso):
    return integrate.quad(integer2, u, o, args=(lam, iso, r, alpha))[0]

def B10EinfachSchraegVor(d, r, lam, alpha, iso):
    if d > r:
        return B10SchraegVor(d, (d - r)/np.cos(alpha), d/np.cos(alpha), r, lam, alpha, iso)
    else: 
        return B10SchraegVor(d, 0., d/np.cos(alpha), r, lam, alpha, iso)
		  
def B10EinfachSchraegRück(d, r, lam, alpha, iso):
      if d > r: 
          return B10SchraegRück(0., r/np.cos(alpha), r, lam, alpha, iso)
      else: 
          return B10SchraegRück(0., d/np.cos(alpha), r, lam, alpha, iso)		  
		
def B10EffSchraegVor(d, lam, eta, iso):
    return 0.94*(B10EinfachSchraegVor(d, 3.16, lam, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.53, lam, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegVor(d, 3.92, lam, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.73, lam, 90. - eta, iso))

def B10EffSchraegRück(d, lam, eta, iso):
    return 0.94*(B10EinfachSchraegRück(d, 3.16, lam, 90. - eta, iso) + B10EinfachSchraegRück(d, 1.53, lam, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegRück(d, 3.92, lam, 90. - eta, iso) + B10EinfachSchraegRück(d, 1.73, lam, 90. - eta, iso)) 

def RestIB10(d, lam, eta, iso):
    return np.exp(-mu(lam,iso)*d/np.cos(90-eta))	

ddrift=1.
dgem=1.1
eta=90.
iso=100.

def Mieze6(lam):
    return B10EffSchraegVor(ddrift, lam, eta, iso) + \
    		RestIB10(ddrift, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
    		RestIB10(ddrift + dgem, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 2*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 3*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 4*dgem, lam, eta, iso) * B10EffSchraegRück(2*ddrift, lam, eta, iso)
                                    	  
def Mieze10(lam):
    return B10EffSchraegVor(ddrift, lam, eta, iso) + \
    		RestIB10(ddrift, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
    		RestIB10(ddrift + dgem, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
    		RestIB10(ddrift + 2*dgem, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 3*dgem, lam, eta, iso) * B10EffSchraegRück(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 4*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 5*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 6*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 7*dgem, lam, eta, iso) * B10EffSchraegVor(dgem, lam, eta, iso) + \
        	RestIB10(ddrift + 8*dgem, lam, eta, iso) * B10EffSchraegRück(2*ddrift, lam, eta, iso)

DetFacpy1=np.vectorize(Mieze6)
DetFacpy2=np.vectorize(Mieze10)

lam=np.linspace(1.8,10,200)
            
py.plot(lam,100*DetFacpy1(lam),'o')
py.plot(lam,100*DetFacpy2(lam),'o')
py.show()