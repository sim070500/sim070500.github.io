import numpy as np
import cupy as cp
import numpy.polynomial.legendre as npl
import time
import os
import tumorfunction as tmf



# parameter setting


split = 2**12
initsigma = 0.01
finalC0 = 8.5
x0 = 400.0
gamma = 0.2
a = 1.25
b = 1.0
Dn = 1.0
Du = 10.0
M = 2.1

parameterdic = np.array([{'a':a, 'b': b, 'Dn':Dn, 'Du':Du, 'M':M, 
                          'finalC0': finalC0, 'x0':x0, 'gamma':gamma}])
np.save('parameter.npy', parameterdic)
# RK4



denotemaxpositive = np.array([])
denotemaxnegative = np.array([])
denotetime = np.array([])
currentvelocity = np.array([0.0])






savefolder = "./Mmodify_a_{}_c_{}_first".format(a, finalC0)
if not os.path.isdir(savefolder):
    os.mkdir(savefolder)




v = np.zeros(split)

order = split
warray =  np.load('./warray_{}.npy'.format(split))

xaxis = np.load('./roots_{}.npy'.format(split))
xaxis = np.append(xaxis[0:split//2], -1*np.flip(xaxis[0:split//2]))

n = np.exp(- (xaxis**2/(2*initsigma**2)))/(initsigma * np.sqrt(2*np.pi))
n = n/max(n)
u = np.full(split, finalC0)
totalsimulationtime = 0


# filenumber = 95000
# totalsimulationtime = filenumber
# n = np.load(savefolder+'/n_{}.npy'.format(filenumber))
# u = np.load(savefolder+'/u_{}.npy'.format(filenumber))
# currentvelocity = np.load(savefolder + '/currentvelocity_{}.npy'.format(filenumber))
# n = np.load('./finaln_eta.npy')
# u = np.load('./finalu_eta.npy')
# currentvelocityvalue = np.load('./finalv.npy')
# currentvelocity = np.array([currentvelocityvalue])
# print(currentvelocity)
# ax.plot(xaxis, n)
# plt.show()
# exit()
v[0:len(xaxis)//2] = -1*currentvelocity[len(currentvelocity)-1]
v[split//2:len(v)] = currentvelocity[len(currentvelocity)-1]

legendrearray = np.load('./legendrearray_{}.npy'.format(split))

dt = 0.001
simulationtime = 0

totaltime = 0
npxaxis = xaxis 
n = cp.array(n)
u = cp.array(u)
v = cp.array(v)
warray = cp.array(warray)
xaxis = cp.array(xaxis)
legendrearray = cp.array(legendrearray)
modenum = cp.arange(len(legendrearray))+0.5
oddindex = cp.array([2*i+1 for i in range(0, len(legendrearray)//2)])
eigvalue1 = cp.arange(0, split+1)
eigvalue2 = -1*eigvalue1*(eigvalue1+1)

npeigvalue1 = np.arange(0, split+1)
npeigvalue2 = -1*npeigvalue1*(npeigvalue1+1)
nochange = 0
prechecktrue = 0


while True:
# while totalsimulationtime<100:
    
    coeffn1 =  tmf.fincoefficient2(n, warray, xaxis, legendrearray, modenum, oddindex)
    coeffu1 =  tmf.fincoefficient2(u, warray, xaxis, legendrearray, modenum, oddindex)
    k1n = tmf.singlelegendreprimefunction(n, xaxis, coeffn1, split, x0, v, legendrearray, eigvalue1) + Dn * tmf.doublelegendreprimefunction(n, xaxis, coeffn1, split, x0, legendrearray, eigvalue2) - M*n*(n-0.5)*(n-1) - gamma * n + a*n*tmf.A(u)
    
    k1u = tmf.singlelegendreprimefunction(u, xaxis, coeffu1, split, x0, v, legendrearray, eigvalue1) + Du * tmf.doublelegendreprimefunction(u, xaxis, coeffu1, split, x0, legendrearray, eigvalue2) - b*n*tmf.A(u)

    coeffn2 =  tmf.fincoefficient2(n+k1n*dt/2, warray, xaxis, legendrearray, modenum, oddindex)
    coeffu2 =  tmf.fincoefficient2(u+k1u*dt/2, warray, xaxis, legendrearray, modenum, oddindex)
    k2n = tmf.singlelegendreprimefunction(n+k1n*dt/2, xaxis, coeffn2, split, x0, v, legendrearray, eigvalue1) + Dn * tmf.doublelegendreprimefunction((n+k1n*dt/2), xaxis, coeffn2, split, x0, legendrearray, eigvalue2) - M*(n+k1n*dt/2)*((n+k1n*dt/2)-0.5)*((n+k1n*dt/2)-1) - gamma * (n+k1n*dt/2) + a*(n+k1n*dt/2)*tmf.A((u+k1u*dt/2))
    k2u = tmf.singlelegendreprimefunction(u+k1u*dt/2, xaxis, coeffu2, split, x0, v, legendrearray, eigvalue1) + Du * tmf.doublelegendreprimefunction((u+k1u*dt/2), xaxis, coeffu2, split, x0, legendrearray, eigvalue2) - b*(n+k1n*dt/2)*tmf.A((u+k1u*dt/2))

    coeffn3 =  tmf.fincoefficient2(n+k2n*dt/2, warray, xaxis, legendrearray, modenum, oddindex)
    coeffu3 =  tmf.fincoefficient2(u+k2u*dt/2, warray, xaxis, legendrearray, modenum, oddindex)
    k3n = tmf.singlelegendreprimefunction(n+k2n*dt/2, xaxis, coeffn3, split, x0, v, legendrearray, eigvalue1) + Dn * tmf.doublelegendreprimefunction((n+k2n*dt/2), xaxis, coeffn3, split, x0, legendrearray, eigvalue2) - M*(n+k2n*dt/2)*((n+k2n*dt/2)-0.5)*((n+k2n*dt/2)-1) - gamma * (n+k2n*dt/2) + a*(n+k2n*dt/2)*tmf.A((u+k2u*dt/2))
    k3u = tmf.singlelegendreprimefunction(u+k2u*dt/2, xaxis, coeffu3, split, x0, v, legendrearray, eigvalue1) + Du * tmf.doublelegendreprimefunction((u+k2u*dt/2), xaxis, coeffu3, split, x0, legendrearray, eigvalue2) - b*(n+k2n*dt/2)*tmf.A((u+k2u*dt/2))

    coeffn4 =  tmf.fincoefficient2(n+k3n*dt, warray, xaxis, legendrearray, modenum, oddindex)
    coeffu4 =  tmf.fincoefficient2(u+k3u*dt, warray, xaxis, legendrearray, modenum, oddindex)
    k4n = tmf.singlelegendreprimefunction(n+k3n*dt, xaxis, coeffn4, split, x0, v, legendrearray, eigvalue1) + Dn * tmf.doublelegendreprimefunction((n+k3n*dt), xaxis, coeffn4, split, x0, legendrearray, eigvalue2) - M*(n+k3n*dt)*((n+k3n*dt)-0.5)*((n+k3n*dt)-1) - gamma * (n+k3n*dt) + a*(n+k3n*dt)*tmf.A((u+k3u*dt))
    k4u = tmf.singlelegendreprimefunction(u+k3u*dt, xaxis, coeffu4, split, x0, v, legendrearray, eigvalue1) + Du * tmf.doublelegendreprimefunction((u+k3u*dt), xaxis, coeffu4, split, x0, legendrearray, eigvalue2) - b*(n+k3n*dt)*tmf.A((u+k3u*dt))    



    dn = dt*(k1n + 2*k2n + 2*k3n + k4n )/6
    du = dt*(k1u + 2*k2u + 2*k3u + k4u )/6

    n = n + dn
    u = u + du
    simulationtime += 1
    totalsimulationtime += 1 
    totaltime += dt

    if nochange < 10:
        if n[split//2] < 10**(-7):

            coeffn =  tmf.fincoefficient2(n, warray, xaxis, legendrearray, modenum, oddindex)
            nnegmax = np.where(n[0:split//2] == max(n))[0]
            startmax = xaxis[nnegmax]
            negmaxposition = tmf.findmax(n.get(), coeffn.get(), npeigvalue1, npeigvalue2, startmax.get())
            
            if len(denotemaxnegative) == 0:
                denotemaxnegative = np.append(denotemaxnegative, negmaxposition)
                denotetime = np.append(denotetime, totaltime)
            elif denotemaxnegative[len(denotemaxnegative)-1] != negmaxposition:
                denotemaxnegative = np.append(denotemaxnegative, negmaxposition)
                denotetime = np.append(denotetime, totaltime)

        if len(denotemaxnegative) !=0 and len(denotemaxnegative) % 5 == 0:
            realnegative = np.arctanh(denotemaxnegative)*x0
            negativevelocity = np.mean(np.diff(realnegative[1:len(realnegative)])/np.diff(denotetime[1:len(denotetime)]))
            positivevelocity = -1*negativevelocity
            v[0:len(v)//2] += negativevelocity
            v[len(v)//2: len(v)] += positivevelocity
            if abs(positivevelocity) < 10**(-5):
                nochange += 1
            currentvelocity = np.append(currentvelocity, currentvelocity[len(currentvelocity)-1] + positivevelocity)
            denotemaxnegative = np.array([])
            denotetime = np.array([])


    if totalsimulationtime % 5000 == 0:

        np.save(savefolder + '/n_{}.npy'.format(totalsimulationtime), n)
        np.save(savefolder + '/u_{}.npy'.format(totalsimulationtime), u)
        np.save(savefolder + '/currentvelocity_{}.npy'.format(totalsimulationtime), currentvelocity)
        checktrue = tmf.checktrue(n, u, v, warray, xaxis, legendrearray, split, x0, Du, b, modenum, oddindex, eigvalue1, eigvalue2)
        # print(checktrue)
        if abs(checktrue-prechecktrue) < 10**(-5) and abs(checktrue) < 10**(-3):
            break
        else: 
            prechecktrue  = checktrue


















