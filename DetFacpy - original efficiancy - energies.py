import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt

# lambda as a function of energy

lam=6. # lambda in AA
li=lam*10**(-10) # incoming wavelength in Angstrom
h_J = 6.626070040e-34 # J*s
hbar=h_J/(2*np.pi)
meV = 1.602176565e-22 # J
mN = 1.674927471e-27 # kg

ki=2*np.pi/li
Ei=hbar**2/(2*mN)*ki**2/meV

def l(energies):
    return np.sqrt(hbar**2/(2*mN)*(2*np.pi)**2/(energies+Ei)/meV)*10**10

# functions from "Efficiency formulas.txt"

na=6.022*10**(23)
rho=2.34
a=10.81
sigma=3837.

def mu(energies, iso):
    return rho * na/a * sigma*iso/100*10**(-24) / 10000*l(energies)/1.8

def Pvor(x, d, r, alpha):
    return (1 - (d - x*np.cos(alpha))/r)/2

def Prück(x, r, alpha):
    return (1 - x*np.cos(alpha)/r)/2

def integer1(x, energies, iso, d, r, alpha):
    return mu(energies, iso)*np.exp(-mu(energies,  iso)*x)*Pvor(x, d, r, alpha)

def integer2(x, energies, iso, r, alpha):
    return mu(energies, iso)*np.exp(-mu(energies,  iso)*x)*Prück(x, r, alpha)

def B10SchraegVor(d, u, o, r, energies, alpha, iso):
    return integrate.quad(integer1, u, o, args=(energies, iso, d, r, alpha))[0]
		
def B10SchraegRück(u, o, r, energies, alpha, iso):
    return integrate.quad(integer2, u, o, args=(energies, iso, r, alpha))[0]
	
def B10EinfachSchraegVor(d, r, energies, alpha, iso):
    if d > r:
        return B10SchraegVor(d, (d - r)/np.cos(alpha), d/np.cos(alpha), r, energies, alpha, iso)
    else: 
        return B10SchraegVor(d, 0., d/np.cos(alpha), r, energies, alpha, iso)
		  
def B10EinfachSchraegRück(d, r, energies, alpha, iso):
      if d > r: 
          return B10SchraegRück(0., r/np.cos(alpha), r, energies, alpha, iso)
      else: 
          return B10SchraegRück(0., d/np.cos(alpha), r, energies, alpha, iso)		  
		
def B10EffSchraegVor(d, energies, eta, iso):
    return 0.94*(B10EinfachSchraegVor(d, 3.16, energies, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.53, energies, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegVor(d, 3.92, energies, 90. - eta, iso) + B10EinfachSchraegVor(d, 1.73, energies, 90. - eta, iso))

def B10EffSchraegRück(d, energies, eta, iso):
    return 0.94*(B10EinfachSchraegRück(d, 3.16, energies, 90. - eta, iso) + B10EinfachSchraegRück(d, 1.53, energies, 90. - eta, iso)) + \
        0.06*(B10EinfachSchraegRück(d, 3.92, energies, 90. - eta, iso) + B10EinfachSchraegRück(d, 1.73, energies, 90. - eta, iso)) 

def RestIB10(d, energies, eta, iso):
    return np.exp(-mu(energies,iso)*d/np.cos(90-eta))	

ddrift=1.
dgem=1.1
eta=90.
iso=100.
                                    	  
def Mieze6(energies):
    return B10EffSchraegVor(ddrift, energies, eta, iso) + \
    		RestIB10(ddrift, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
    		RestIB10(ddrift + dgem, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 2*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 3*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 4*dgem, energies, eta, iso) * B10EffSchraegRück(2*ddrift, energies, eta, iso)
            
def Mieze10(energies):
    return B10EffSchraegVor(ddrift, energies, eta, iso) + \
    		RestIB10(ddrift, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
    		RestIB10(ddrift + dgem, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
    		RestIB10(ddrift + 2*dgem, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 3*dgem, energies, eta, iso) * B10EffSchraegRück(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 4*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 5*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 6*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 7*dgem, energies, eta, iso) * B10EffSchraegVor(dgem, energies, eta, iso) + \
        	RestIB10(ddrift + 8*dgem, energies, eta, iso) * B10EffSchraegRück(2*ddrift, energies, eta, iso)            

energies=np.linspace(-2.2,100,100)

DetFacpy1=np.vectorize(Mieze6)
DetFacpy2=np.vectorize(Mieze10)

# approximation from Andreas Wendl
def DetFacpy3(energies):
	return (6*(0.0233+0.079*(0.5*np.log10(81.82) - 0.5*np.log10(81.92/lam**2 + energies)))-1)+1

plt.plot(energies,100*DetFacpy1(energies),'o')
plt.plot(energies,100*DetFacpy2(energies),'o')
plt.plot(energies,100*DetFacpy3(energies),'-')
plt.show()