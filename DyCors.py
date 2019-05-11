import numpy as np
import time
from math import log

#from incompNS import *
#from airfoilNS import *
from airfoil2E import *


def LatinHyperCube(d,m):
    # Latin hypercube sampling
    #   d: dimension
    #   m: number of sample points
    delta = np.ones(d)/m
    X     = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            X[i,j] = (2*i+1)/2*delta[j]
    P = np.zeros((m,d),dtype=int)

    P[:,0] = np.arange(m)
    if m%2 == 0:
        k      = m//2
    else:
        k      = (m-1)//2
        P[k,:] = (k+1)*np.ones((1,d))
    for j in range(1,d):
        P[0:k,j] = np.random.permutation(np.arange(k))
        for i in range(k):
            if np.random.random() < 0.5:
                P[m-1-i,j] = m-1-P[i,j]
            else:
                P[m-1-i,j] = P[i,j]
                P[i,j]     = m-1-P[i,j]
    IPts = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            IPts[i,j] = X[P[i,j],j]
    return IPts



def surrogateRBF(x,f):
    # build a surrogate surface using cubic RBF's + linear polynomial
    #   (x,f): high-dimensional points and associated costs f(x)
    n,d = x.shape
    # RBF-matrix
    R   = -2*np.dot(x,x.T) + np.sum(x**2,axis=1) + np.sum((x.T)**2,axis=0)[:,np.newaxis]
    Phi = np.sqrt(abs(R))**3             # RBF-part
    P   = np.hstack((np.ones((n,1)),x))  # polynomial part
    Z   = np.zeros((d+1,d+1))            # zero matrix
    A   = np.block([[Phi,P],[P.T,Z]])    # patched together
    # right-hand side
    F     = np.zeros(n+d+1)
    F[:n] = f                            # rhs-vector
    return np.linalg.solve(A,F)          # solution


def phi(n,n0,Nmax,d):
    # strictly decreasing probability function
    p0  = min(20/d,1)
    phi = p0*(1 - log(n-n0+1)/log(Nmax-n0+1))
    return phi


def genTrials(p,pslct,sig,bL,bU):
    # perturb SOME coordinates of BEST population member
    nN,nV = np.shape(p)
    M     = 0
    while np.sum(np.sum(M)) == 0:                        # enusre at least one coord perturbed
        M = 1*(np.random.rand(nN,nV)<=pslct)                # take those below threshold
    pM    = p + M*np.random.normal(0,sig*(bU-bL),(nN,nV))   # move the points, M is a 0-1-mask
    pM    = np.clip(pM,bL,bU)                               # enforce bounds
    return pM


def evalRBF(x,s,y):
    # evaluate the surrogate surface at {y}
    #   x: RBF-points
    #   s: coefficient vector of surrogate surface
    #   y: trial points
    m,d = y.shape
    # RBF-matrix (evaluated at {y})
    R   = -2*np.dot(x,y.T) + np.sum(y**2,axis=1) + np.sum(x**2,axis=1)[:,np.newaxis]
    Phi = np.sqrt(abs(R.T))**3           # RBF-part
    P   = np.hstack((np.ones((m,1)),y))  # polynomial part
    A   = np.block([Phi,P])              # patched together
    return np.dot(A,s)                   # evaluation


def selectNext(p,f,pP):
    # RBF score
    imin, imax = f.argmin(), f.argmax() # min and max trial point indices
    fmin, fmax = f[imin], f[imax]       # min and max function values
    Vrbf = (f - fmin)/(fmax-fmin)       # RBF score

    # distance score
    R = -2*np.dot(pP,p.T) + np.sum(p**2,axis=1) + np.sum(pP**2,axis=1)[:,np.newaxis]
    d = np.min(np.sqrt(abs(R.T)),axis=1)# distance of trial points to nearest population
    imin, imax = d.argmin(), d.argmax() # min and max distance indices
    dmin, dmax = d[imin], d[imax]       # min and max distance values
    Vdist = (dmax - d)/(dmax-dmin)      # distance score

    # overall score and best point
    wR = np.random.uniform()            # randomly select RBF weighting
    wD = 1-wR                           # find corresponding distance weighting
    V = wR*Vrbf + wD*Vdist              # calculate overall score
    pnext = p[V.argmin(),:]             # find best point
    return pnext


def updateBest(pnext,fnext,pB,fB,Cs,Cf):
    # Update best point and success/failure counters
    if fnext < fB:                          # if success
        print('Success, updating best coordinate')
        Cs, Cf = Cs+1, 0                    # increment success, reset fail
        pB, fB = pnext, fnext               # update best point
    else:                                   # if not
        Cs, Cf = 0, Cf+1                    # reset success, increment fail
    return pB,fB,Cs,Cf


def updateStep(sig,sigmin,Cs,Cf,Ts,Tf):
    # Update stepsize if successes/failures have reached thresholds
    if Cs >= Ts:                            # if success = threshold
        sig, Cs = 2*sig, 0                  # double stepsize, reset success
        print('Multiple successes, increasing stepsize to', sig)
    elif Cf >= Tf:                          # if failures = threshold
        sig, Cf = max(sig/2,sigmin), 0      # halve stepsize, reset fail
        print('Multiple failures, reducing stepsize to', sig)
    return sig,Cs,Cf


def DyCors(nP,nV,bL,bU):
    # nP: population size
    # nV: number of decision (unknwon) variables
    # bL: lower bound of decision variables
    # bU: upper bound of decision variables

    # simulated annealing parameters
    Nmax   = 100            # max no. of function evaluations
    nNei   = 1000           # no. of neighbors
    sig    = 0.1            # stepsize
    sigmin = 1.e-5          # min stepsize
    Cs,Cf  = 0,0            # success and failure counters
    Ts,Tf  = 3,max(nV,5)    # success and failure thresholds

    # initialization
    n = nP
    pP    = bL +(bU-bL)*LatinHyperCube(nV,nP) # LatinHyperCube for initial population
    print('\nPerforming initial',nP,'function evaluations')
    fP    = np.apply_along_axis(solveNS,1,pP) # f() - function evaluation
    print('Initial population complete')
    ib    = fP.argmin()                       # determine best solution
    pB,fB = pP[ib,:],fP[ib]
    

    # DyCors main loop
    while n < Nmax:
        # evaluate surrogate surface
        s     = surrogateRBF(pP,fP)             # fit response surface sn using all points
        pslct = phi(n,nP,Nmax,nV)               # prob of perturbing coord

        # retry selecting next point until it differs from current best
        pnext = pB
        while np.array_equal(pnext,pB):
            # generate trial points about pB & evaluate of surrogate surface
            pnew  = np.outer(nNei*[1],pB)           # duplicate nNei times
            pnew  = genTrials(pnew,pslct,sig,bL,bU) # mutate (shake)
            fnew  = evalRBF(pP,s,pnew)              # evalutae new trial points

            # select next point to perform full function evaluation
            pnext = selectNext(pnew,fnew,pP)  # determine trial point to run full sim


        # perform next function evaluation
        print('\nPerforming function evaluation',n+1,'at',pnext)
        fnext = solveNS(pnext)
        print('cost \t\t=',fnext)

        # Update ready for next iteration
        pB,fB,Cs,Cf = updateBest(pnext,fnext,pB,fB,Cs,Cf)  # counters & best point
        sig,Cs,Cf = updateStep(sig,sigmin,Cs,Cf,Ts,Tf)         # stepsize

        # Add new point to population
        pP = np.vstack((pP,pnext[np.newaxis,:]))
        fP = np.concatenate((fP,np.array([fnext])))


        n += 1
        print('best cost \t=',fB)
        
    # Write data to file for post processing
    f = open("NSLDv3.txt","w+")
    for i in range(np.shape(pP)[0]):
        for j in range(np.shape(pP)[1]):
            f.write('%f' %(pP[i][j]))
            f.write('\t')
        f.write('%f' %(fP[i]))
        f.write('\n')
    f.write('\n')
    f.close()
    
    return pB,fB,pP,fP


if __name__ == '__main__':
    # Bounds for airfoilNS
    #bL, bU = np.array([0.05,0,0.3,0]), np.array([0.15,0.08,0.8,np.pi/6])

    # Bounds for airfoil2E
    bL = np.array([0.05,    0, 0.3,       0,     0.05,    0, 0.3,       0, -0.1, 0.01]) 
    bU = np.array([0.15, 0.08, 0.8, np.pi/6,     0.15, 0.08, 0.8, np.pi/4,  0.1,  0.1])
    
    nV = len(bL)       # number of variables (dimensions, coordinates)
    nP = 3*nV          # size of initial population
    

    pB,fB,pP,fP  = DyCors(nP,nV,bL,bU) # Commence DyCors algorithm
    print('\n\nOptimisation complete')
    print('Global minimum',pB)
    print('Function value at minimum',fB)
