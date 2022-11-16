import numpy as np
import os
import numpy.polynomial.legendre as npl
import cupy as cp



def A(u):
    ans = ((1+np.tanh(u-2))/2) - ( 1 + np.tanh(0-2) )/2
    return ans 

def invA(value):
    print((value +( 1 + np.tanh(0-2) )/2)*2 - 1)
    ans = np.arctanh((value +( 1 + np.tanh(0-2) )/2)*2 - 1)+2
    return ans

def findlagendreweight(split):
    if os.path.isfile('root_{}.npy'.format(split)):
        Broots = np.load('roots_{}.npy'.format(split))
    else:
        A = np.zeros(split+1)
        A[-1] = 1
        B = npl.Legendre(A)
        Broots = B.roots()
        np.save('roots_{}.npy'.format(split), Broots)

    now = np.zeros(split+1)
    now[-1] = 1
    after = np.zeros(split+1)
    after[-2] = 1
    pder = split*(Broots*npl.legval(Broots, now)-npl.legval(Broots, after))/(Broots**2 -1)
    warray = 2/((1-Broots**2)*(pder**2))
    np.save('warray_{}.npy'.format(split), warray)

    return warray 

def fincoefficient2(function, warray, xaxis, legendmatrix, modenum, oddindex):
    
    oldfunc = function * warray
    finalcoeff = cp.dot(legendmatrix, oldfunc)
    finalcoeff[oddindex] = 0
    finalcoeff = finalcoeff*modenum
    
    return finalcoeff

def doublelegendreprimefunction( function, xaxis, coeff, split, x0, legendmatrix, eigvalue2):
    
    data = cp.dot(coeff*eigvalue2, legendmatrix)
    data = data*(1-xaxis**2)/(x0**2)

    return data

def singlelegendreprimefunction( function, xaxis, coeff, split, x0, velocity, legendmatrix, eigvalue1):
    alson = -1*velocity*cp.dot(coeff*eigvalue1, legendmatrix)*xaxis/x0
    nminuscoeff = cp.zeros(len(eigvalue1))
    nminuscoeff[0:len(nminuscoeff)-1] = coeff[1:len(coeff)]*eigvalue1[1:len(eigvalue1)]
    nminus = velocity*cp.dot(nminuscoeff, legendmatrix)/x0
    finalprime = alson + nminus
    
    return finalprime

def npfincoefficient2(function, warray, xaxis, legendmatrix, modenum, oddindex):
    
    oldfunc = function * warray
    finalcoeff = np.dot(legendmatrix, oldfunc)
    finalcoeff[oddindex] = 0
    finalcoeff = finalcoeff*modenum
    
    return finalcoeff

def npdoublelegendreprimefunction( function, xaxis, coeff, split, x0, legendmatrix, eigvalue2):
    
    data = np.dot(coeff*eigvalue2, legendmatrix)
    data = data*(1-xaxis**2)/(x0**2)

    return data

def npsinglelegendreprimefunction( function, xaxis, coeff, split, x0, velocity, legendmatrix, eigvalue1):
    alson = -1*velocity*np.dot(coeff*eigvalue1, legendmatrix)*xaxis/x0
    nminuscoeff = np.zeros(len(eigvalue1))
    nminuscoeff[0:len(nminuscoeff)-1] = coeff[1:len(coeff)]*eigvalue1[1:len(eigvalue1)]
    nminus = velocity*np.dot(nminuscoeff, legendmatrix)/x0
    finalprime = alson + nminus
    
    return finalprime


def findmax( function, coeff, eigvalue1, eigvalue2,  startx = 0):
    x = startx
    alson = coeff*eigvalue1
    nminuscoeff = np.zeros(len(eigvalue1))
    nminuscoeff[0:len(nminuscoeff)-1] = coeff[1:len(coeff)]*eigvalue1[1:len(eigvalue1)]
    while True:
        single = npl.legval(x, alson)*x/(x**2 -1) + npl.legval(x, nminuscoeff)*(-1)/(x**2-1)
        if abs(single) < 10**(-12):
            break

        # minus in eig2
        double = single*2*x/(1-x**2) + npl.legval(x, coeff*eigvalue2)/(1-x**2)
        dx = single/double
        x -= dx
    return x


def checktrue(n, u, v, warray, xaxis, legendrearray, split, x0, Du, b, modenum, oddindex, eigvalue1, eigvalue2):
    
    coeffu = fincoefficient2(u, warray, xaxis, legendrearray, modenum, oddindex)
    single = singlelegendreprimefunction(u, xaxis, coeffu, split, x0, v, legendrearray, eigvalue1) 

    double = doublelegendreprimefunction(u, xaxis, coeffu, split, x0, legendrearray, eigvalue2)
    predictn = (single + Du*double)/(b*A(u))
    return np.max(abs(predictn-n))
    
