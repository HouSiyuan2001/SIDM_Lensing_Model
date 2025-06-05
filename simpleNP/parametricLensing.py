import numpy as np
from scipy.integrate import quad
import math

########### Cosmology
# Constants
h = 0.7
Omegam0 = 0.3
Omegab0 = 0.05
Omegar0 = 0
Omegak0 = 0
OmegaL0 = 1 - Omegam0
H0 = h / 9.777752  # Gyr^-1
G = 4.30093e-6 # kpc/Msun (km/sec)^2 

# Function definitions
def Ez(z):
    return np.sqrt(OmegaL0 + Omegak0 * (1 + z)**2 + Omegam0 * (1 + z)**3 + Omegar0 * (1 + z)**4)

def Hz(z):
    return H0 * Ez(z)

def tL(z):
    integrand = lambda zx: 1 / (Hz(zx) * (1 + zx))
    result, _ = quad(integrand, 0, z)
    return result

DHubble = (3 * 10**5) / H0  # Hubble distance in kpc

def DC(z):
    integrand = lambda zx: 1 / Ez(zx)
    result, _ = quad(integrand, 0, z)
    return DHubble * result  # l.o.s comoving distance

arcsec = (1 / 3600) * (1 / 360) * 2 * np.pi

def dA(z):
    return DC(z) / (1 + z)  # Angular diameter distance

def DLS(zl, zs):
    integrand = lambda zx: 1 / Ez(zx)
    result, _ = quad(integrand, zl, zs)
    return DHubble * result / (1 + zs)

########## Parametric model

def fited_a(x):
    a, b, c, d, g, h, i, j, k = 1.11323951e+00, -1.26183924e+00,  2.87611831e+00, -3.61577534e+00,2.77768241e+00, -4.15935892e-01,  1.35746091e-02, -3.86240111e-02,-2.63332737e-04
    return a + b*x**0.1 + c*x**0.2 + d*x**0.5 + g*x**0.9+h*x**2+i*x**12+j*x**20+k*x**87

def fited_b(x):
    a, b, c, d, g, h, i, j,k,l = 6.59669818e+00,  2.21289181e+00, -9.56284544e+00,  9.10385757e+00, -4.01289646e+00,  3.26789716e+00,  2.07641520e+00,  1.18377949e-01, -2.10519668e-04,  4.21396052e-07
    return a + b*x**0.01 + c*x**0.2 + d*x**0.6 + g*x**0.9+h*x**2+i*x**12+j*x**51+k*x**109+l*x**203

def fited_c(x):
    a, b, c, d, g, h, i, j = 1.79309631e+00,  5.10780441e-01, -1.43007683e+00,  1.39244251e+00, -7.85787233e-01,  1.60384916e-01,  3.14558126e-02,  7.14758319e-04
    return a + b*x**0.1 + c*x**0.2 + d*x**0.4 + g*x**0.9+h*x**2+i*x**12+j*x**67

def fited_p(x):
    a, b, c, d, g, h, i, j = 6.94266549, 4.96669252, -12.52779021, 7.54562512, 3.21238241, 2.86297753, -0.32646638, 0.30926922
    return a + b*x**0.2 + c*x**0.3 + d*x**0.8 + g*x**2+h*x**11+i*x**22+j*x**37

def fited_s(x):
    a, b, c, d, g, h, i, j = 1.82544721e+00, -5.99111491e-01, 8.44850940e-01, -4.34593883e-01, 8.17550753e-02, 1.67462383e-02, -1.75109111e-03, 3.82847738e-03
    return a + b*x**0.2 + c*x**0.3 + d*x**0.8 + g*x**2+h*x**11+i*x**22+j*x**37

def fit_alpha(r, a, b, c, d, e):
    term1 = (a * c * (b + 2 * c * r) * np.log(b * r + c * r**2 + 1)**(c - 1)) / (b * r + c * r**2 + 1)
    term2 = (d * e * np.log(d * r + 1)**(e - 1)) / (d * r + 1)
    derivative = term1 - term2
    return derivative
