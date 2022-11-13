# Program:
# 	This program simulates a traffic system with identical drivers.
#   Also, I use the theoretical eignevector as initial condition.
#   By doing so, we can find the theoretical eigenvalue as soon as possible, and check the coefficient for phase boundary that the real part of the eigenvalue transit from negative to positive.
#   The description of the program is under construction.
# _______________________________________________________________

# import module

import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import pandas as pd
import os
import shutil
import function as myf
import simulation_function as myf_simulate
# _______________________________________________________________

# Declare variable
# 
# totalcarn : Total number of vihicles
# length : tThe length of the circular track
# amplitude : The amplitude of the initial perturbation
# alpha : The equal-spacing of veicles at steady states.

totalcarn = 2**5
length = totalcarn*5
amplitude = 0.001
alpha = 2*np.pi/totalcarn 
fig, ax = plt.subplots()

# Variables for finding eigenstates 
#
# deletetime: Iteration number for deleting data. As the record data reach the size of deletetime, the index [0:deletetime//2]in the record metrix will be deleted to avoid a tremendous matrix.
#
# uptime: A least time for applying linear stability analysis. A simulation might explode when a system is unstable. Thus, a simulation will be terminated under this circumstance. In this condition, the stability of a system will be detemined by its fourier amplitude value. Moreover, to accelerate the simulation, when the amplitude is smaller than 10**(-6), a system is considered to be return to its sterady state. 
#
# unstable_delta_coefficient: The change step of reaction time for finding the threshold value when the system is unstable.

deletetime = 20000
uptime = 3000000
unstable_delta_coefficient = 0.02


# Create a veriable 'finaldata' to denote results
# variable 'final_sense' denotes all simulations results of predicted threshould value
if os.path.isfile('./finaldata.npy'):
    finaldata = np.read('./finaldata.npy')
else: 
    finaldata = np.array([{'sigma':[], 'a':[], 'initial_sense':[]}])
    final_sense = finaldata[0]['a']
# _______________________________________________________________

# Main Program

# Declar the sigma range that we want to simulate
simulation_sigma = np.linspace(0, 0.22, 16)
wmean = 1

# One hundrad realization of Gaussian random field is applied in the simulation.
while len(final_sense) < 100:
    print(len(final_sense))

    # record_coefficient: record the results of finished reaction time
    # simulation_coefficient: 
    # Initially we set the condition that the system is not near its threshold(fixed point I defined)
    # The individual difference can be tuned by changin the simulation_coefficient, which is the index of the matrix simulation_sigma.
    # 'initvalue2' is used to record initial value
    # 'initial_sense' is the theoretial threshold


    record_coefficient = []
    is_near_fixed_point = False
    simulation_coefficient = 0
    w = myf.creatw(totalcarn , 1, simulation_sigma[simulation_coefficient])

    fftw = myf.fft_function(totalcarn, w)/len(w)
    initvalue = myf.xinit(amplitude, alpha, length, totalcarn, w)
    initvalue2 = myf.xinit(amplitude, alpha, length, totalcarn, w)
    f = initvalue[2]
    [eigenvalue, eigenfunction] = myf.eigenvalue(w, totalcarn)
    initial_sense = myf.find_boundary(eigenvalue, f)
    np.save('wfile.npy', w)

    
    # 'denotesense' and 'denoteslope' are use to record the dynamic of the first fourier mode. A slope is measured by collecting all peak values of the first fourier mode amplitude. The matrix 'denoteslope' denotes the slope value and 'denotesense' denotes the correponding sense
    denotesense = []
    denoteslope = []


    # Define the inital difference between the theoretical threshold and the paramters we use, sense = initial_sense +coeffcient. 
    # paramter 'how_much_time_reach_another_phase' is used to refine results.

    coefficient = 0.005
    delta_coefficient = coefficient/2
    how_much_time_reach_another_phase = 0
    previous_type = 'I do nothing'
    testtime = 0
    
    while True:
        try:
            os.makedirs('./{}'.format(coefficient))
        except:
            shutil.rmtree('./{}'.format(coefficient))
            os.makedirs('./{}'.format(coefficient))
        record_coefficient.append(coefficient)
        savetime = 0
        sense = initial_sense + coefficient
        z = (-sense+np.sqrt(sense**2+4*sense*f*eigenvalue))/2
        originx = initvalue2[0]
        originv = initvalue2[1]

        # We set initial condition along the eigenvector to compare theoretial result and numerical results

        eigendirection1 = []
        eigendirection2 = []
        eigneindex1 = np.where(np.real(z)==max(np.real(z)))[0][0]
        finaleigenvalue = (z[np.where(np.real(z)==max(np.real(z)))])[0]
        for index in range(0, len(eigenvalue)):
            if index != eigneindex1:
                if abs(np.real(z[index])-np.real(finaleigenvalue))<10**(-8):
                    eigneindex2 = index
        for index in range(0, len(eigenfunction)):
            eigendirection1.append(eigenfunction[index][eigneindex1])
            eigendirection2.append(eigenfunction[index][eigneindex2])

        eigendirection1 = np.array(eigendirection1)
        eigendirection2 = np.array(eigendirection2)
        eigendirection1 = np.roll(eigendirection1, totalcarn//2-1)
        eigendirection2 = np.roll(eigendirection2, totalcarn//2-1)
        firsteigenvector = amplitude*np.append(np.array([0]), eigendirection1 + eigendirection2)
        initx = myf.fft_function(totalcarn, (firsteigenvector)*totalcarn, 'i')
        initx = np.real((initx))
        vfirsteigenvector = amplitude*np.append(np.array([0]), finaleigenvalue*eigendirection1 + np.conjugate(finaleigenvalue)*eigendirection2)
        initv = myf.fft_function(totalcarn, (vfirsteigenvector)*totalcarn, 'i')
        initv = np.real((initv))
        x = initvalue[0]+initx 
        v = initvalue[1]+initv
        y = initx

        totaltime = 0
        time = []
        denotex = []
        denoteflowtime = []
        yplot = []
        yplotheory = []
        tmax = []
        tmin = []
        meanvalue = []
        meanvaluetime = []
        simulationtime = 0
        plottime = 0
        theoryy = myf.fft_function(totalcarn, totalcarn*amplitude*np.append(np.array([0]), eigendirection1*np.exp(finaleigenvalue*totaltime) + eigendirection2*np.exp(np.conjugate(finaleigenvalue)*totaltime)), 'i')
        yplot.append(np.real(myf.fft_function(totalcarn, y)[1])/len(myf.fft_function(totalcarn, y)))
        yplotheory.append(np.real(myf.fft_function(totalcarn, theoryy)[1]/len(myf.fft_function(totalcarn, y))))

        time.append(totaltime)
        ydot = initv

    # ===================================================

    # Step:
    #   (1) Run simulation until we contour ten maximun and minimun
    #   (2) Calculate slope according to the data we record
    #   (3) If there is no maxiun, we need to direct denote slope and the typ of stability

        # (1)
        while True:
            print(simulationtime)
            run_nonlinear = myf_simulate.animate_real_nonlinear(w, originx, x, v, length, totalcarn, sense, simulationtime, RK4=True)
            [x, v, y, deltax, dt] = run_nonlinear
            totaltime += dt
            theoryy = myf.fft_function(totalcarn, totalcarn*amplitude*np.append(np.array([0]), eigendirection1*np.exp(finaleigenvalue*totaltime) + eigendirection2*np.exp(np.conjugate(finaleigenvalue)*totaltime)), 'i')
            
            if simulationtime % 100 == 0 and simulationtime >= 0:
                yplot.append(np.real(myf.fft_function(totalcarn, y)[1])/len(myf.fft_function(totalcarn, y)))
                yplotheory.append(np.real(myf.fft_function(totalcarn, theoryy)[1]/len(myf.fft_function(totalcarn, y))))
                time.append(totaltime)
            simulationtime += 1
            if totaltime//50000 > savetime:
                np.save('./{}/{}_x'.format(coefficient,totaltime), x)
                np.save('./{}/{}_v'.format(coefficient,totaltime), v)
                savetime += 1

            if len(time) >= deletetime:

                for i in range(2, len(yplot)-1):
                    if yplot[i] > yplot[i+1] and yplot[i] > yplot[i-1] and yplot[i] > 0:
                        if time[i] not in meanvaluetime:
                            meanvalue.append(np.log(abs(yplot[i])))
                            meanvaluetime.append(time[i])
                if True:
                    ax.cla()
                    ax.plot(time, yplot, 'r.')
                    ax.plot(time, yplotheory, 'b.')
                    fig.savefig('./{}/test{}.png'.format(coefficient, simulationtime))
                    ax.cla()
                    ax.plot(meanvaluetime, meanvalue, 'r.')
                    fig.savefig('./{}/mean{}.png'.format(coefficient, simulationtime))
                    plottime += 1
                del time[0:deletetime//2]
                del yplot[0:deletetime//2]
                del yplotheory[0:deletetime//2]


                if np.log10(abs(myf.fft_function(totalcarn, y))[1]/len(myf.fft_function(totalcarn, y))) < -6:
                    if totaltime < uptime:
                        style = False
                        current_type = 'stable'
                        break
                    else:
                        style = True
                        slope = 0
                        for slopeindex in range(len(meanvalue)-10, len(meanvalue)-1):
                            slope += (meanvalue[slopeindex+1]-meanvalue[slopeindex])/(meanvaluetime[slopeindex+1]-meanvaluetime[slopeindex])
                            slope = slope/(len(meanvalue)-1)
                        break
                elif np.log10(abs(myf.fft_function(totalcarn, y))[1]/len(myf.fft_function(totalcarn, y))) > -1.2:
                    if totaltime < uptime:
                        style = False
                        current_type = 'unstable'
                        break
                    else:
                        style = True
                        slope = 0
                        for slopeindex in range(len(meanvalue)-10, len(meanvalue)-1):
                            slope += (meanvalue[slopeindex+1]-meanvalue[slopeindex])/(meanvaluetime[slopeindex+1]-meanvaluetime[slopeindex])
                            slope = slope/(len(meanvalue)-1)
                        break
            if len(meanvalue) > 10:
                style = True
                slope = np.diff(np.array(meanvalue)[-len(meanvalue)//2-1:])/np.diff(np.array(meanvaluetime)[-len(meanvaluetime)//2-1:])
                slope = np.mean(slope)
                break

        ax.cla()
        ax.plot(meanvaluetime, meanvalue, 'r.')

    # ===================================================
    #  Step: Evaluate the stability of the system
    #      (1) If slope > 0 denote it as instable by 'rX'
    #      (2) If slope < 0 denote it as stable by 'bo'
    #
        # if len(yplotmax) or len(meanvalue) != 0:
        if style:
            if len(meanvalue) != 0:
                if slope < 0:
                    current_type = 'stable'

                elif slope > 0:
                    current_type = 'unstable'

            denoteslope.append(slope)
            denotesense.append(sense)
    # ===================================================
    #
    # Step:
    #   (1) Check whether we get the boundary
    #   (2) If not, change the coefficient

        if previous_type == 'stable' and current_type == 'stable':
            previous_type = current_type
            if np.all(abs(np.array(record_coefficient)-(coefficient-delta_coefficient)) > 0.9*delta_coefficient ):
                coefficient -= delta_coefficient
            else:
                if how_much_time_reach_another_phase < 4:
                    delta_coefficient = delta_coefficient/2
                    coefficient -= delta_coefficient
                else:
                    break
        elif previous_type == 'unstable' and current_type == 'unstable':
            previous_type = current_type
            if np.all(abs(np.array(record_coefficient)-(coefficient+delta_coefficient)) > 0.9*delta_coefficient ):
                coefficient += delta_coefficient
            else:
                if how_much_time_reach_another_phase < 4:
                    delta_coefficient = delta_coefficient/2
                    coefficient += delta_coefficient
                else:
                    break

        elif previous_type == 'stable' and current_type == 'unstable':
            is_near_fixed_point = True
            if how_much_time_reach_another_phase < 4:
                delta_coefficient = delta_coefficient/2
                previous_type = current_type
                coefficient += delta_coefficient
                how_much_time_reach_another_phase += 1
            else:
                break

        elif previous_type == 'unstable' and current_type == 'stable':
            if how_much_time_reach_another_phase < 4:
                delta_coefficient = delta_coefficient/2
                previous_type = current_type
                coefficient -= delta_coefficient
                how_much_time_reach_another_phase += 1
            else:
                break

        if previous_type == 'I do nothing':
            previous_type = current_type
            if current_type == 'stable':
                coefficient -= delta_coefficient
            elif current_type == 'unstable':
                coefficient += unstable_delta_coefficient

        testtime += 1
        if testtime >= 4:
            break

    # ===================================================
    #   Step: Calculate the slope and calculate the zero point
    #       (1) Calculate slope
    #       (2) Calaulate the zero point
    #       (3) Renew the final data
    senseslope = 0

    for datasense in range(0, len(denotesense)-1):
        senseslope += (denoteslope[datasense+1]-denoteslope[datasense])/(denotesense[datasense+1]-denotesense[datasense])
    senseslope = senseslope/(len(denoteslope)-1)

    b0 = 0
    for datasense in range(0, len(denotesense)):
        b0 += denoteslope[datasense]-senseslope * denotesense[datasense]
    b0 = b0 / len(denotesense)
    final_sense_cal = -b0/senseslope 

    # Add this result to the database

    finaldata[0]['sigma'].append(np.sqrt(np.var(w)))
    finaldata[0]['a'].append(final_sense_cal)
    finaldata[0]['initialsense'].append(initial_sense)



