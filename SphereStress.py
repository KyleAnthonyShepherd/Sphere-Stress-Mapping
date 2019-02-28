'''
Stresses in a sphere compressed between rigid platens
Model from
HIRAMATSYU, Y., and OKA, Y., 1966, Int. J . Rock Mech. Min. Sci., 3, 89.

With corrections from
Václav Pejchal, Goran Žagar, Raphaël Charvet, Cyril Dénéréaz, Andreas Mortensen,
Compression testing spherical particles for strength: Theory of the meridian crack test and implementation for microscopic fused quartz,
Journal of the Mechanics and Physics of Solids,
Volume 99,
2017,
Pages 70-92,
ISSN 0022-5096,
https://doi.org/10.1016/j.jmps.2016.11.009.

This code implements the mathematical model of HIRAMATSYU and OKA. It models a
hard sphere being compressed between two stiff platens. It determines the stress
distribution within the sphere.
The platen has a spherical indents of radius a, so the force is distributed on the sphere

###
Code Written by:
Kyle Shepherd, at Oak Ridge National Laboratory
kas20@rice.edu
May 21, 2018
###
'''
#### Import BLock ####
# the import block imports needed modules, and spits out a json file with
# version numbers so the code can be repeatable
file = open("ModuleVersions.json", 'w')
modules = {}

import os
import itertools

import sys
modules['Python'] = dict([('version', sys.version_info)])

import json
modules['json'] = dict([('version', json.__version__)])

import numpy
modules['numpy'] = dict([('version', numpy.__version__)])
from numpy.polynomial.legendre import legval as Leg

json.dump(modules, file, indent=4, sort_keys=True)
file.close()
#### END Import Block ####

def DiskCordGeneration(density,tol):
    '''
This function creates a uniform distrubution of points on the disk, to be
sampled or calculated.
It generates points at a constant angle with a constant spacing
Additional points are added when two angles diverge far enough from each other

points are spaced from angle .0001 to pi/2
angle 0 causes divide by zero issues in the calculations
points are spaced from radius 0 to 1-tol
the calculations have gibbs instability at radius=1, so points are created some
distance tol from the edge of the disk for numeric stability

Inputs:
###
density: the spacing between points
format: float
Example: 0.05
smaller values increase the number of sample points

tol: defines the distance between the disk edge and the edge points
Format: float
example: .005
###

The code outputs a tuple containing the angles and the radius
example: (angles,r)
angles is a list of values
r is a list of lists,
where r[n] contains a list of radi corresponding to the nth angle
'''
    # create initial points, at the borders of the disk
    angles=[]
    r=[]
    angles.append(.0001)
    r.append(numpy.append(numpy.arange(0,1-tol,density),1-tol))
    angles.append(numpy.pi/2)
    r.append(numpy.append(numpy.arange(density,1-tol,density),1-tol))

    # loop adding angles until there is no more space
    CandidateAngle = numpy.pi/4 # first angle is in between the first two
    loop=0
    while 1==1:
        # calculates the radius when the newer angle can place a point
        r0=density/CandidateAngle
        if r0>1:
            break # end loop if the calculated radius exceeds 1
        # creates new points starting ar r0 for all the new angles.
        for arc in range(0,2**loop):
            angles.append(numpy.pi/2-CandidateAngle-CandidateAngle*2*arc)
            r.append(numpy.append(numpy.arange(r0,1-tol,density),1-tol))
        # at each angle addition, the angle is divided by 2
        CandidateAngle=CandidateAngle/2
        loop=loop+1
    return (angles,r)



def Calcs(a,v,MaxIter,density,rtol,vtol,Verbose=True):
    '''
This function performs the stress calculations on the compressed sphere.
It calls the DiskCordGeneration to get points, and calculates the stresses
at each angle, using vectorized operations.

Inputs:
###
a: the radius of the indenter
format: float
Example: 0.1

v: The poisson ratio of the sphere
Format: float
example: .33

MaxIter: The maximum number of iterations to perform.
The calculations use Legendre Polynomials, and more iterations will be more accurate
Format: Integer
example: 1000
If this value is set too low, and the rtol value is set too low, inaccurate
results at the disk edge may be obtained.

density: the spacing between points
format: float
Example: 0.05
smaller values increase the number of sample points

rtol: defines the distance between the disk edge and the edge points
format: float
example: .005

vtol: Defines the tolerance of the iterative calculations
As each Legendre Polynomial is added, the result will change by some value. If
the change of all the values along an angle is less than this number, the
answer is deemed accurate and is outputed
format: float
example: .000001

Verbose: A switch to print diagnostic values to the console
Prints the iteration loop that it breaks on, and prints the number of angles left
to calculate
format: Bool
example: True
###

The code outputs a tuple containing the angles and radius of the sample points,
and the 3x3 stress matrix at each sample point.
example: (ang,r,StressTensor)
ang is a list of angle values
r is a list of radius values,
StressTensor is a list of 3x3 matricies, where StressTensor[n] is the stress
tensor at ang[n] and radius [n]
'''
    # gets sample points for calculations
    (angles,rlist)=DiskCordGeneration(density,rtol)

    R=1 # unit sphere
    alpha = numpy.arccos(1-2*(a/R)**2)/2 # sets alpha value, size of the platen contact area

    # initialize values, legendre polynomials
    Palpha=numpy.zeros(MaxIter)
    PalphaM1=numpy.zeros(MaxIter)
    PPrimeAlpha=numpy.zeros(MaxIter)

    # precalc Leg, legendre polynomials, values at alpha, iterative formula
    LegVals=[]
    LegVals.append(1)
    x=numpy.cos(alpha)
    LegVals.append(x)
    for n in range(2,2*MaxIter+1):
        LegVals.append((2*n-1)/n*x*LegVals[n-1]-(n-1)/n*LegVals[n-2])


    # precalc Palpha value used in calculations
    for i in range(1,MaxIter):
        Palpha[i] = LegVals[2*i]
        PalphaM1[i] = LegVals[2*i-1]
        PPrimeAlpha[i] = (2*i)/(numpy.cos(alpha)**2-1) * (numpy.cos(alpha)*Palpha[i]-PalphaM1[i])

    # initialize list to store Stress matrix values
    StressTensor = [[] for i in range(len(angles))]
    k=0 # index for StressTensor list storage

    # loop over each sample angle
    for z,ang in enumerate(angles):
        r=rlist[z] # get the list of radius points for the given angle

        # get initial values of equations A1, A2, A3, and A4
        SigmaTheta = -.5+numpy.zeros(len(r))
        SigmaPhi = -.5+numpy.zeros(len(r))
        SigmaR = -.5+numpy.zeros(len(r))
        Shear = 0+numpy.zeros(len(r))

        # checks change in value as the calculation is iterativerly performed
        SigmaPhiold=numpy.array(SigmaPhi)

        # precalc Leg values at ang, iterative formula
        LegVals=[]
        LegVals.append(1)
        x=numpy.cos(ang)
        LegVals.append(x)
        for n in range(2,2*MaxIter+1):
            LegVals.append((2*n-1)/n*x*LegVals[n-1]-(n-1)/n*LegVals[n-2])

        # loop over the iterative formula
        # each iteration is a better and better apporximation, using higher order
        # legendre polynomials
        # start iteration at
        for i in range(1,MaxIter):

            # A1 and A2 term 1
            c1 = ( (-(4*i+1)*(i-1)+(4*i+1)**2*v)/(2*i*(4*i**2+2*i+1)+2*i*(4*i+1)*v) )*(r/R)**(2*i)
            # A1 and A2 term 2
            c2 = ( ((4*i+1)*(4*i**2+4*i-1)+2*(4*i+1)*v)/(2*(4*i**2-1)*(4*i**2+2*i+1)+2*(4*i+1)*(4*i**2-1)*v) )*(r/R)**(2*i-2)
            # A1 and A2 term 3
            c3 = 1+numpy.cos(alpha)
            # A1 and A2 term 4
            c4 = ( (-(4*i+1)*(2*i+5)+4*(4*i+1)*v)/(4*i*(2*i+1)*(4*i**2+2*i+1)+4*i*(2*i+1)*(4*i+1)*v) )*(r/R)**(2*i)
            # A1 and A2 term 5
            c5 = ( ((4*i+1)*(4*i**2+4*i-1)+2*(4*i+1)*v)/(4*i*(4*i**2-1)*(4*i**2+2*i+1)+4*i*(4*i+1)*(4*i**2-1)*v) )*(r/R)**(2*i-2)

            # get P, legendre polynomial, terms
            Pang = LegVals[2*i]
            PangM1 = LegVals[2*i-1]
            PangM2 = LegVals[2*i-2]
            # and get the derivatives
            dPang = (2*i)/(numpy.sin(ang)) * (numpy.cos(ang)*Pang-PangM1)
            dPangM1 = (2*i)/(numpy.sin(ang)) * (numpy.cos(ang)*PangM1-PangM2)
            d2Pdang = -(2*i)/(numpy.tan(ang)*numpy.sin(ang)) * (numpy.cos(ang)*Pang-PangM1) + (2*i)/(numpy.sin(ang)) * (-numpy.sin(ang)*Pang+numpy.cos(ang)*dPang-dPangM1)

            # calculate A1 and A2, adding to previous terms
            SigmaTheta = SigmaTheta -(1/2)*((c1+c2)*c3*PPrimeAlpha[i]*Pang+(c4+c5)*c3*PPrimeAlpha[i]*d2Pdang)
            SigmaPhi = SigmaPhi -(1/2)*((c1+c2)*c3*PPrimeAlpha[i]*Pang+(c4+c5)*c3*PPrimeAlpha[i]*dPang/numpy.tan(ang))

            # A3 term 1
            c1 = (2*(1+v)*(1-2*v)*(4*i+1)*(numpy.cos(alpha)*Palpha[i]-PalphaM1[i]))
            # A3 term 2
            c2 = ((8*i**2+8*i+3)*(2*v)+(8*i**2+4*i+2)*(1-2*v))
            # A3 term 3
            c3 = ((4*i**2-2*i-3)*v)/((1+v)*(1-2*v))*(r/R)**(2*i)
            # A3 term 4
            c4 = (2*i+1)*(2*i-2)/(2*(1+v))*(r/R)**(2*i)
            # A3 term 5
            c5 = (4*i**2*(2*i+2)*v)/((2*i+1)*(1+v)*(1-2*v))*(r/R)**(2*i-2)
            # A3 term 6
            c6 = (2*i*(4*i**2+4*i-1))/(2*(2*i+1)*(1+v))*(r/R)**(2*i-2)

            # calculate A3, adding to previous terms
            SigmaR = SigmaR - (1/2)*(1/(1-numpy.cos(alpha)))*(c1/c2)*(c3+c4-c5-c6)*Pang

            # A4 term 1
            c1 = -(4*i+1)*(4*i**2+4*i-1)-2*(4*i+1)*v\
            # A4 term 2
            c2 = 4*i*(2*i+1)*(4*i**2+2*i+1)+4*i*(2*i+1)*(4*i+1)*v
            # A4 term 3
            c3 = (r/R)**(2*i)-(r/R)**(2*i-2)
            # A4 term 4
            c4 = 1+numpy.cos(alpha)

            # calculate A4, adding to previous terms
            Shear = Shear -(1/2)*(c1/c2)*c3*c4*PPrimeAlpha[i]*dPang

            # ends calculation iterations when the change is less than vtol
            test=numpy.abs(SigmaPhi-SigmaPhiold)
            if numpy.max(test) < vtol:
                if Verbose:
                    print(i)
                break

            # update the value to compare to next iteration
            SigmaPhiold=numpy.array(SigmaPhi)
        if Verbose:
            print(len(angles)-z)

        # save values to StressTensor
        for i in range(0,len(r)):
            StressTensor[k].append(stress)
        k=k+1 # update index for next angle

    # flatten the angle, r, and stress list of lists to a single list
    ang=[ [angles[i]]*len(rlist[i]) for i in range(len(rlist))]
    ang=list(itertools.chain(*ang))
    r=list(itertools.chain(*rlist))
    StressTensor=numpy.array(list(itertools.chain(*StressTensor)))

    return (ang,r,StressTensor)

def StressPlotter(Title,ColorLevels,Label,ang,r,Stress,ShowSamplePoints=True):

    ### things to import
    import matplotlib
    matplotlib.use('Agg') # no UI backend
    import matplotlib.pyplot as plt
    ###

    ### set up plotting area
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    dx=7.5
    dy=6.5
    fig = plt.figure(figsize=(dx, dy))
    plot = fig.add_axes([.5/dx,.5/dy,5/dx,5/dy],projection='polar')

    Stress=numpy.array(Stress)

    if ShowSamplePoints==True:
        plot.plot(ang,r,'.',markersize=.5,markeredgewidth=0.0,color=[1,0,0])
    ax=plot.tricontourf(ang,r,Stress,levels=ColorLevels)

    plot.set_theta_zero_location("N")
    plot.set_theta_direction(-1)
    plot.set_thetalim(0,numpy.pi/2)
    ###

    ### add labels and save fig
    plot.set_title(Title,pad=40)
    colorbar=fig.add_axes([6.25/dx,.5/dy,.5/dx,5/dy])
    fig.colorbar(ax,cax=colorbar,ticks=ColorLevels)
    colorbar.set_ylabel(Label)
    RadiusShow=fig.add_axes([.5/dx,6/dy,a*5/dx,.1/dy])
    RadiusShow.patch.set_color('b')
    RadiusShow.xaxis.set_visible(False)
    RadiusShow.yaxis.set_visible(False)
    fig.savefig(Title+'.png',dpi=1000)  #savefig, don't show
    ###

if __name__ == "__main__":
    R=1 # radius of sphere
    a=.1 # radius of indentation
    v=.33 #poisson ratio
    MaxIter = 7000
    rtol=.01
    vtol=.00001
    density=.01
    (ang,r,StressTensor)=Calcs(a,v,MaxIter,density,rtol,vtol)

    Title='Tension, Compressive Radius = '+str(a)+', Poisson Ratio = '+str(v)
    Label='Normalized Tensile First Principal Stress Distribution'
    ColorLevels=numpy.linspace(0,1,num=30)
    Tension=[]
    for stress in StressTensor:
        Pstress = numpy.linalg.eig(stress)
        Tension.append(numpy.max(Pstress[0]))
    Tension=numpy.array(Tension)
    Tension[numpy.where(Tension<0)]=0
    ColorLevels=numpy.linspace(0,numpy.max(Tension),num=30)
    StressPlotter(Title,ColorLevels,Label,ang,r,Tension)

    Title='Vertical Stress, Compressive Radius = '+str(a)+', Poisson Ratio = '+str(v)
    Label='Normalized Vertical Stress Distribution'
    ColorLevels=numpy.linspace(0,1,num=30)
    CompressionV=[]
    for k,stress in enumerate(StressTensor):
        # CoordChange = numpy.array([[numpy.sin(ang[k]),numpy.cos(ang[k]),0],[0,0,1],[numpy.cos(ang[k]),-numpy.sin(ang[k]),0]])
        CoordChange = numpy.array([[numpy.sin(ang[k]),numpy.cos(ang[k]),0],[0,0,1],[numpy.cos(ang[k]),-numpy.sin(ang[k]),0]])
        CStress = CoordChange.dot(stress).dot(numpy.transpose(CoordChange))
        CompressionV.append(CStress[1][1])
    CompressionV=numpy.array(CompressionV)
    ColorLevels=numpy.linspace(numpy.min(CompressionV),0,num=30)
    StressPlotter(Title,ColorLevels,Label,ang,r,CompressionV)

    Title='Vertical Strain, Compressive Radius = '+str(a)+', Poisson Ratio = '+str(v)
    Label='Normalized Vertical Strain Distribution'
    ColorLevels=numpy.linspace(0,1,num=30)
    D=1/((1+v)*(1-2*v))*numpy.array([   [1-v,v,v,0,0,0],
                                        [v,1-v,v,0,0,0],
                                        [v,v,1-v,0,0,0],
                                        [0,0,0,(1-2*v)/2,0,0],
                                        [0,0,0,0,(1-2*v)/2,0],
                                        [0,0,0,0,0,(1-2*v)/2]])
    Dinv=numpy.linalg.inv(D)
    ElasticV=[]
    for k,stress in enumerate(StressTensor):
        CoordChange = numpy.array([[numpy.sin(ang[k]),numpy.cos(ang[k]),0],[0,0,1],[numpy.cos(ang[k]),-numpy.sin(ang[k]),0]])
        CStress = CoordChange.dot(stress).dot(numpy.transpose(CoordChange))
        Elastic = Dinv.dot([CStress[0,0],CStress[1,1],CStress[2,2],CStress[0,1],CStress[1,2],CStress[0,2]])
        # CStress = CoordChange.dot(stress).dot(numpy.transpose(CoordChange))
        ElasticV.append(Elastic[2])
    ElasticV=numpy.array(ElasticV)
    nang=numpy.array(ang)
    print(numpy.sum(nang[numpy.where(nang==numpy.pi/2)]*ElasticV[numpy.where(nang==numpy.pi/2)]))
    ColorLevels=numpy.linspace(numpy.min(ElasticV),numpy.max(ElasticV),num=30)
    StressPlotter(Title,ColorLevels,Label,ang,r,ElasticV,ShowSamplePoints=False)
