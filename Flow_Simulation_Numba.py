#GPU WILL NOT LET YOU USE LISTS. USE TUPLES


# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:45:17 2020

@author: stefan
"""
from __future__ import division

import numpy as np
import random
import numba
from numba import float64, int32, cuda, void
import math 
import inline_try as inl


REAL_R = 'f4'
REAL_X = 'f8'
DOUBLES = True

if DOUBLES:
    REAL = 'f8'
    COMPLEX = 'c16'
else:
    REAL = 'f4'
    COMPLEX = 'c8'

UINT = 'u4'
TARGET= 'parallel'
#TARGET = 'cuda'
inline_kw = dict(inline='always', fastmath=True, cache=True)
vec_kw = dict(nopython=True, target=TARGET)

def replace_types(s):
    return s.replace('f?', REAL).replace('c?', COMPLEX).replace('u?', UINT)

real = np.dtype(REAL).type


#X varible, has all the stuff
#X: [:,0-2] xyz
#X: [:,3]   dt
#X: [:,4-6] nx ny nz
#X: [:,7]   ds



''' Remap: is designed to deal with the common problem of dense local stretching.
For a simple vortex and dye placement the vortex will wrap quicker and quicker
around the tip of the bight, resulting in large stretching around a small point.
This is clearly a numerical nightmare so we resolve it by absorbing neighboring
ds lengths and normalizng the tangents to unit length. '''

@numba.guvectorize([replace_types("f?[:,:], u?, u?, f?[:]")], '(np, nd), (), () -> (nd)', **vec_kw)
def remap(X, i, samples, Xo):
    Xi = X[i]
    weight = real(1./samples)
    Xr0, Xr1, Xr2 = inl.xyz(Xi)
    Xr3 = inl.dt(Xi)
    
    Xr4, Xr5, Xr6 = inl.norm(inl.nxyz(Xi))
    Xr7 = real(0.)
    
    if i < (len(X) - 1):
        Ti = inl.sv3_mul(Xi[7], inl.nxyz(Xi))
        Xf = X[i + 1]
        D = inl.v3_sub(inl.xyz(Xf), inl.xyz(Xi))
        Tf = inl.sv3_mul(Xi[7], inl.nxyz(Xf))
        
        
        for j in range(0, samples):
            t = real(j)/real(samples)
            
            '''Simpsons rule: endpoints get half the weight.'''
            if (j == 0) or (j == samples):
                w = real(0.5 * weight)
            else:
                w = weight
             
                
            Xr7 = w * inl.mag( inl.sv3_mul((t-1)*(real(3)*t - real(1)), Ti) + inl.sv3_mul((real(6) - real(6)*t)*t, D) + inl.sv3_mul(t*(real(3)*t - real(2)), Tf))
            
            
            
    
    '''There is probably a much better way to write this:'''
    #Xo[0], Xo[1], Xo[2], Xo[3], Xo[4], Xo[5], Xo[6], Xo[7] = Xr0, Xr1, Xr2, Xr3, Xr4, Xr5, Xr6, Xr7
    Xo[0] = Xr0
    Xo[1] = Xr1
    Xo[2] = Xr2
    Xo[3] = Xr3
    Xo[4] = Xr4
    Xo[5] = Xr5
    Xo[6] = Xr6
    Xo[7] = Xr7
      
    return
#indices = np.arange(len(X), dtype=UINT)







'''Enough numerical constants to make a grown man cry.  They're only used for the 
RK45 method in the integrators.'''
C2 = real( 1. / 5. )
C3 = real( 3. / 10. )
C4 = real( 4. / 5. )
C5 = real( 8. / 9. )
C6 = real( 1. )
C7 = real( 1. )

A21 = real( 1. / 5. )

A31 = real( 3. / 40. )
A32 = real( 9. / 40. )

A41 = real( 44. / 45. )
A42 = real(-56. / 15. )
A43 = real( 32. / 9. )

A51 = real( 19372. / 6561. )
A52 = real(-25360. / 2187. )
A53 = real( 64448. / 6561. )
A54 = real(-212. / 729. )

A61 = real( 9017. / 3168. )
A62 = real(-355. / 33. )
A63 = real( 46732. / 5247. )
A64 = real( 49. / 176. )
A65 = real(-5103. / 18656. )

A71 = real( 35. / 384. )
A72 = real( 0. )
A73 = real( 500. / 1113. )
A74 = real( 125. / 192. )
A75 = real(-2187. / 6784. )
A76 = real( 11. / 84. )

E1 = real( 71. / 57600. )
E2 = real( 0. )
E3 = real(-71. / 16695. )
E4 = real( 71. / 1920. )
E5 = real(-17253. / 339200. )
E6 = real( 22. / 525. )
E7 = real(-1. / 40. )






''' A utility funciton. call this to GU the inline calculate_line_U  '''
@numba.guvectorize([replace_types("f?[:], f?[:]")], "(nd) -> (nd)", **vec_kw)
def line_U(X, U):
    U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7] = inl.calculate_line_U(X)
    
    

@numba.guvectorize([replace_types("f?[:], f?, f?, f?, f?[:]")], "(nd ),(),(),() -> (nd)", **vec_kw)
def line_one_step(Xi, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7])
    
    j = 0.
    h = min(max(real(inl.dt(Xi)), hmin), dt)
    tr = dt
    next_allow_retry = True
    k1 = inl.calculate_line_U(X)
    
    while j < 1.:
        '''When the next point will land near a timestep point we want to
               instead land directly on it.  Sometimes we will cheat a bit
               to avoid landing really close and having to do a tiny timestep.'''
        if h > (tr * 0.9):
            h = tr
             
        else:
            h = h
            
        
        k2 = inl.calculate_line_U(inl.v8_add(X, inl.sv8_mul(h, inl.sv8_mul(A21, k1))))
        k3 = inl.calculate_line_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A31, k1), inl.sv8_mul(A32, k2)))))
        k4 = inl.calculate_line_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A41, k1), inl.v8_add(inl.sv8_mul(A42, k2), inl.sv8_mul(A43, k3))))))
        k5 = inl.calculate_line_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A51, k1), inl.v8_add(inl.sv8_mul(A52, k2), inl.v8_add(inl.sv8_mul(A53, k3), inl.sv8_mul(A54, k4)))))))
        
        k6 = inl.calculate_line_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A61, k1), inl.v8_add(inl.sv8_mul(A62, k2), inl.v8_add(inl.sv8_mul(A63, k3), inl.v8_add(inl.sv8_mul(A64, k4), inl.sv8_mul(A65, k5))))))))
        
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7 = inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A71, k1), inl.v8_add(inl.sv8_mul(A72, k2), inl.v8_add(inl.sv8_mul(A73, k3), inl.v8_add(inl.sv8_mul(A74, k4), inl.v8_add(inl.sv8_mul(A75, k5), inl.sv8_mul(A76, k6))))))))
        
        k7 = inl.calculate_line_U((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7))
        
        err = inl.v8_add(inl.sv8_mul(E1, k1), inl.v8_add(inl.sv8_mul(E2, k2), inl.v8_add(inl.sv8_mul(E3, k3), inl.v8_add(inl.sv8_mul(E4, k4), inl.v8_add(inl.sv8_mul(E5, k5), inl.v8_add(inl.sv8_mul(E6, k6), inl.sv8_mul(E7, k7)))))))


        error = inl.mag(inl.xyz(err))
        delta = real(0.84)*pow((tol/error), real(0.2))
        
        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max( h*min(max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr = tr - h
            h = max( h*min( max(delta, 0.1), 4.0), hmin)
            next_allow_retry = True
            
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7] = inl.v8_add(Xn, inl.sv8_mul(tr, k7))
                
                j = j + 1.
                
            else:
                k1 = k7
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
                next_allow_retry = True
                
    return



@numba.guvectorize([replace_types("f?[:], f?[:,:], f?[:]")], "(nd),(a, b) -> (nd)", **vec_kw)
def lines_U(X, R, U):
    U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7] = inl.calculate_lines_U(X, R)



'''This is our integrator.  It implements a RK45 method to calculate the trajectory of a point using our
velocity calcualtor calculate_vortex_U which can be found in inline_try.py   This function also implements
a changing time step to account for near-vortex points with high velocities that need closer inspeciton.'''
@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", **vec_kw)
#@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", nopython=True, target='cuda')
def lines_one_step(Xi, R, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    tr = dt
    next_allow_retry = True
    
    k1 = inl.calculate_lines_U(X, R)
    
    while j < 1.:
        #When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.
        if h > tr*0.9:
            h = tr
            
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because it's written in the OpenCL form, not fit for GUvectorize, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        k2 = inl.calculate_lines_U(inl.v8_add(X, inl.sv8_mul(h, inl.sv8_mul(A21, k1))), R)
       
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        k3 = inl.calculate_lines_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A31, k1), inl.sv8_mul(A32, k2)))), R)
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        k4 = inl.calculate_lines_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A41, k1), inl.v8_add(inl.sv8_mul(A42, k2), inl.sv8_mul(A43, k3))))), R)
       
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        k5 = inl.calculate_lines_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A51, k1), inl.v8_add(inl.sv8_mul(A52, k2), inl.v8_add(inl.sv8_mul(A53, k3), inl.sv8_mul(A54, k4)))))), R)
       
       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        k6 = inl.calculate_lines_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A61, k1), inl.v8_add(inl.sv8_mul(A62, k2), inl.v8_add(inl.sv8_mul(A63, k3), inl.v8_add(inl.sv8_mul(A64, k4), inl.sv8_mul(A65, k5))))))), R)
       
       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7 = inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A71, k1), inl.v8_add(inl.sv8_mul(A72, k2), inl.v8_add(inl.sv8_mul(A73, k3), inl.v8_add(inl.sv8_mul(A74, k4), inl.v8_add(inl.sv8_mul(A75, k5), inl.sv8_mul(A76, k6))))))))
       
        
        k7 = inl.calculate_lines_U((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7), R)
        
        
        err = inl.v8_add(inl.sv8_mul(E1, k1), inl.v8_add(inl.sv8_mul(E2, k2), inl.v8_add(inl.sv8_mul(E3, k3), inl.v8_add(inl.sv8_mul(E4, k4), inl.v8_add(inl.sv8_mul(E5, k5), inl.v8_add(inl.sv8_mul(E6, k6), inl.sv8_mul(E7, k7)))))))

        
        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))


        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7] = inl.v8_add(Xn, inl.sv8_mul(tr, k7))
              
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
    
    return

@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", **vec_kw)
#@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", nopython=True, target='cuda')
def lines_one_step_T(Xi, R, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7], Xi[8], Xi[9], Xi[10], Xi[11], Xi[12], Xi[13], Xi[14], Xi[15])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    tr = dt
    next_allow_retry = True
    
    k1 = inl.calculate_lines_U_T(X, R)
    
    while j < 1.:
        #When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.
        if h > tr*0.9:
            h = tr
            
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because it's written in the OpenCL form, not fit for GUvectorize, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        k2 = inl.calculate_lines_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.sv16_mul(A21, k1))), R)
       
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        k3 = inl.calculate_lines_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A31, k1), inl.sv16_mul(A32, k2)))), R)
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        k4 = inl.calculate_lines_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A41, k1), inl.v16_add(inl.sv16_mul(A42, k2), inl.sv16_mul(A43, k3))))), R)
       
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        k5 = inl.calculate_lines_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A51, k1), inl.v16_add(inl.sv16_mul(A52, k2), inl.v16_add(inl.sv16_mul(A53, k3), inl.sv16_mul(A54, k4)))))), R)
       
       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        k6 = inl.calculate_lines_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A61, k1), inl.v16_add(inl.sv16_mul(A62, k2), inl.v16_add(inl.sv16_mul(A63, k3), inl.v16_add(inl.sv16_mul(A64, k4), inl.sv16_mul(A65, k5))))))), R)
       
       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A71, k1), inl.v16_add(inl.sv16_mul(A72, k2), inl.v16_add(inl.sv16_mul(A73, k3), inl.v16_add(inl.sv16_mul(A74, k4), inl.v16_add(inl.sv16_mul(A75, k5), inl.sv16_mul(A76, k6))))))))
       
        
        k7 = inl.calculate_lines_U_T((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15), R)
        
        
        err = inl.v16_add(inl.sv16_mul(E1, k1), inl.v16_add(inl.sv16_mul(E2, k2), inl.v16_add(inl.sv16_mul(E3, k3), inl.v16_add(inl.sv16_mul(E4, k4), inl.v16_add(inl.sv16_mul(E5, k5), inl.v16_add(inl.sv16_mul(E6, k6), inl.sv16_mul(E7, k7)))))))

        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))


        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7], Xf[8], Xf[9], Xf[10], Xf[11], Xf[12], Xf[13], Xf[14], Xf[15] = inl.v16_add(Xn, inl.sv16_mul(tr, k7))
              
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
    
    return




@numba.guvectorize([replace_types("f?[:], f?[:,:], f?[:]")], "(nd),(a, b) -> (nd)", **vec_kw)
def vortex_U(X, R, U):
    U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7] = inl.calculate_vortex_U(X, R)



'''This is our integrator.  It implements a RK45 method to calculate the trajectory of a point using our
velocity calcualtor calculate_vortex_U which can be found in inline_try.py   This function also implements
a changing time step to account for near-vortex points with high velocities that need closer inspeciton.'''
@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", **vec_kw)
def vortex_one_step(Xi, R, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    tr = dt
    next_allow_retry = True
    
    k1 = inl.calculate_vortex_U(X, R)
    
    while j < 1.:
        #When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.
        if h > tr*0.9:
            h = tr
            
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because it's written in the OpenCL form, not fit for GUvectorize, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        k2 = inl.calculate_vortex_U(inl.v8_add(X, inl.sv8_mul(h, inl.sv8_mul(A21, k1))), R)
       
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        k3 = inl.calculate_vortex_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A31, k1), inl.sv8_mul(A32, k2)))), R)
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        k4 = inl.calculate_vortex_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A41, k1), inl.v8_add(inl.sv8_mul(A42, k2), inl.sv8_mul(A43, k3))))), R)
       
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        k5 = inl.calculate_vortex_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A51, k1), inl.v8_add(inl.sv8_mul(A52, k2), inl.v8_add(inl.sv8_mul(A53, k3), inl.sv8_mul(A54, k4)))))), R)
       
       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        k6 = inl.calculate_vortex_U(inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A61, k1), inl.v8_add(inl.sv8_mul(A62, k2), inl.v8_add(inl.sv8_mul(A63, k3), inl.v8_add(inl.sv8_mul(A64, k4), inl.sv8_mul(A65, k5))))))), R)
       
       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7 = inl.v8_add(X, inl.sv8_mul(h, inl.v8_add(inl.sv8_mul(A71, k1), inl.v8_add(inl.sv8_mul(A72, k2), inl.v8_add(inl.sv8_mul(A73, k3), inl.v8_add(inl.sv8_mul(A74, k4), inl.v8_add(inl.sv8_mul(A75, k5), inl.sv8_mul(A76, k6))))))))
       
        
        k7 = inl.calculate_vortex_U((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7), R)
        
        
        err = inl.v8_add(inl.sv8_mul(E1, k1), inl.v8_add(inl.sv8_mul(E2, k2), inl.v8_add(inl.sv8_mul(E3, k3), inl.v8_add(inl.sv8_mul(E4, k4), inl.v8_add(inl.sv8_mul(E5, k5), inl.v8_add(inl.sv8_mul(E6, k6), inl.sv8_mul(E7, k7)))))))

        
        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))


        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7] = inl.v8_add(Xn, inl.sv8_mul(tr, k7))
              
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7)
    
    return




# INTEGRATOR FOR THE STRAIN TENSOR.  BASICALLY MAKE EVERYTHING WORK FOR 16 COMPONENTS, NOT 8 LIKE THE ABOVE.
@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),() -> (nd)", **vec_kw)
def vortex_one_step_T(Xi, R, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7], Xi[8], Xi[9], Xi[10], Xi[11], Xi[12], Xi[13], Xi[14], Xi[15])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    tr = dt
    next_allow_retry = True
    
    k1 = inl.calculate_vortex_U_T(X, R)
    
    while j < 1.:
        #When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.
        if h > tr*0.9:
            h = tr
            
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because it's written in the OpenCL form, not fit for GUvectorize, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        k2 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.sv16_mul(A21, k1))), R)
       
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        k3 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A31, k1), inl.sv16_mul(A32, k2)))), R)
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        k4 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A41, k1), inl.v16_add(inl.sv16_mul(A42, k2), inl.sv16_mul(A43, k3))))), R)
       
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        k5 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A51, k1), inl.v16_add(inl.sv16_mul(A52, k2), inl.v16_add(inl.sv16_mul(A53, k3), inl.sv16_mul(A54, k4)))))), R)
       
       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        k6 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A61, k1), inl.v16_add(inl.sv16_mul(A62, k2), inl.v16_add(inl.sv16_mul(A63, k3), inl.v16_add(inl.sv16_mul(A64, k4), inl.sv16_mul(A65, k5))))))), R)
       
       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A71, k1), inl.v16_add(inl.sv16_mul(A72, k2), inl.v16_add(inl.sv16_mul(A73, k3), inl.v16_add(inl.sv16_mul(A74, k4), inl.v16_add(inl.sv16_mul(A75, k5), inl.sv16_mul(A76, k6))))))))
       
        
        k7 = inl.calculate_vortex_U_T((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15), R)
        
        
        err = inl.v16_add(inl.sv16_mul(E1, k1), inl.v16_add(inl.sv16_mul(E2, k2), inl.v16_add(inl.sv16_mul(E3, k3), inl.v16_add(inl.sv16_mul(E4, k4), inl.v16_add(inl.sv16_mul(E5, k5), inl.v16_add(inl.sv16_mul(E6, k6), inl.sv16_mul(E7, k7)))))))

        
        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))


        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7], Xf[8], Xf[9], Xf[10], Xf[11], Xf[12], Xf[13], Xf[14], Xf[15] = inl.v16_add(Xn, inl.sv16_mul(tr, k7))
              
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
    
    return



@numba.guvectorize([replace_types("f?[:], f?[:,:], f?[:,:], f?, f?, f?, f?[:]")], "(nd),(nR, rd),(nR, rd),(),(),() -> (nd)", **vec_kw)
def two_vortex_one_step_T(Xi, R1, R2, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7], Xi[8], Xi[9], Xi[10], Xi[11], Xi[12], Xi[13], Xi[14], Xi[15])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    #h = hmin
    tr = dt
    next_allow_retry = True
    
    k1 = inl.v16_add(inl.calculate_vortex_U_T(X, R1), inl.calculate_vortex_U_T(X, R2))
    
    while j < 1.:
        #When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.
        if h > tr*0.9:
            h = tr
            
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because it's written in the OpenCL form, not fit for GUvectorize, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        X2 = inl.v16_add(X, inl.sv16_mul(h, inl.sv16_mul(A21, k1)))
        k2 =  inl.v16_add(inl.calculate_vortex_U_T(X2, R1), inl.calculate_vortex_U_T(X2, R2))
        
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        X3 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A31, k1), inl.sv16_mul(A32, k2))))
        k3 =  inl.v16_add(inl.calculate_vortex_U_T(X3, R1), inl.calculate_vortex_U_T(X3, R2))
        
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        X4 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A41, k1), inl.v16_add(inl.sv16_mul(A42, k2), inl.sv16_mul(A43, k3)))))
        k4 =  inl.v16_add(inl.calculate_vortex_U_T(X4, R1), inl.calculate_vortex_U_T(X4, R2))
        
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        X5 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A51, k1), inl.v16_add(inl.sv16_mul(A52, k2), inl.v16_add(inl.sv16_mul(A53, k3), inl.sv16_mul(A54, k4))))))
        k5 = inl.v16_add(inl.calculate_vortex_U_T(X5, R1), inl.calculate_vortex_U_T(X5, R2))

       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        X6 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A61, k1), inl.v16_add(inl.sv16_mul(A62, k2), inl.v16_add(inl.sv16_mul(A63, k3), inl.v16_add(inl.sv16_mul(A64, k4), inl.sv16_mul(A65, k5)))))))
        k6 = inl.v16_add(inl.calculate_vortex_U_T(X6, R1), inl.calculate_vortex_U_T(X6, R2))

       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A71, k1), inl.v16_add(inl.sv16_mul(A72, k2), inl.v16_add(inl.sv16_mul(A73, k3), inl.v16_add(inl.sv16_mul(A74, k4), inl.v16_add(inl.sv16_mul(A75, k5), inl.sv16_mul(A76, k6))))))))
               
        k7 = inl.v16_add(inl.calculate_vortex_U_T((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15), R1), inl.calculate_vortex_U_T((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15), R2))

        err = inl.v16_add(inl.sv16_mul(E1, k1), inl.v16_add(inl.sv16_mul(E2, k2), inl.v16_add(inl.sv16_mul(E3, k3), inl.v16_add(inl.sv16_mul(E4, k4), inl.v16_add(inl.sv16_mul(E5, k5), inl.v16_add(inl.sv16_mul(E6, k6), inl.sv16_mul(E7, k7)))))))

        
        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))


        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7], Xf[8], Xf[9], Xf[10], Xf[11], Xf[12], Xf[13], Xf[14], Xf[15] = inl.v16_add(Xn, inl.sv16_mul(tr, k7))
                
              
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
    
    return




@numba.guvectorize([replace_types("f?[:], f?[:,:], f?, f?, f?, f?, f?[:]")], "(nd),(nR, rd),(),(),(),() -> (nd)", **vec_kw)
def map_point(Xi, R, num_crossings, dt, tol, hmin, Xf):
    X = (Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], Xi[7], Xi[8], Xi[9], Xi[10], Xi[11], Xi[12], Xi[13], Xi[14], Xi[15])
    
    j = 0.
    h = min( max(inl.dt(X), hmin), dt)
    tr = dt
    next_allow_retry = True
    
    k1 = inl.calculate_vortex_U_T(X, R)
    
    while j < 1.:
        '''When the next point will land near a timestep point we want to
            #instead land directly on it.  Sometimes we will cheat a bit
            #to avoid landing really close and having to do a tiny timestep.'''
        
   
        '''RK45 right here.  The notation is an unfortunate product of some short-sighted inline functions written before the rest of 
        this simulation.   The commented line above each k# is FOR UNDERSTANDING AND CONVENIENCE ONLY.  Uncommenting these lines will
        brick the program because they're written in the OpenCL form, but they are easier to read'''
       #k2 = calculate_vortex_U(X + h * (A21*k1), R, nR);
        k2 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.sv16_mul(A21, k1))), R)
       
       #k3 = calculate_vortex_U(X + h * (A31*k1 + A32*k2), R, nR);
        k3 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A31, k1), inl.sv16_mul(A32, k2)))), R)
        
       #k4 = calculate_vortex_U(X + h * (A41*k1 + A42*k2 + A43*k3), R, nR);
        k4 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A41, k1), inl.v16_add(inl.sv16_mul(A42, k2), inl.sv16_mul(A43, k3))))), R)
       
       #k5 = calculate_vortex_U(X + h * (A51*k1 + A52*k2 + A53*k3 + A54*k4), R, nR);
        k5 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A51, k1), inl.v16_add(inl.sv16_mul(A52, k2), inl.v16_add(inl.sv16_mul(A53, k3), inl.sv16_mul(A54, k4)))))), R)
       
       #k6 = calculate_vortex_U(X + h * (A61*k1 + A62*k2 + A63*k3 + A64*k4 + A65*k5), R, nR)
        k6 = inl.calculate_vortex_U_T(inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A61, k1), inl.v16_add(inl.sv16_mul(A62, k2), inl.v16_add(inl.sv16_mul(A63, k3), inl.v16_add(inl.sv16_mul(A64, k4), inl.sv16_mul(A65, k5))))))), R)
       
       #Xn = X + h * (A71*k1 + A72*k2 + A73*k3 + A74*k4 + A75*k5 + A76*k6);
        Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15 = inl.v16_add(X, inl.sv16_mul(h, inl.v16_add(inl.sv16_mul(A71, k1), inl.v16_add(inl.sv16_mul(A72, k2), inl.v16_add(inl.sv16_mul(A73, k3), inl.v16_add(inl.sv16_mul(A74, k4), inl.v16_add(inl.sv16_mul(A75, k5), inl.sv16_mul(A76, k6))))))))
       
        
        k7 = inl.calculate_vortex_U_T((Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15), R)
        
        
        err = inl.v16_add(inl.sv16_mul(E1, k1), inl.v16_add(inl.sv16_mul(E2, k2), inl.v16_add(inl.sv16_mul(E3, k3), inl.v16_add(inl.sv16_mul(E4, k4), inl.v16_add(inl.sv16_mul(E5, k5), inl.v16_add(inl.sv16_mul(E6, k6), inl.sv16_mul(E7, k7)))))))

        
        
        error = inl.mag(inl.xyz(err))
        
        delta = real(0.84)*pow((tol/error), real(0.2))



        if next_allow_retry == True and error > tol and h > hmin:
            next_allow_retry = False
            h = max(h * min( max(delta, real(0.01)), real(4.0)), hmin)
            
        else:
            tr -= h
            h = max(h * min( max(delta, real(0.1)), real(4.0)), hmin)
            next_allow_retry = True
            
            '''The successful condition'''
            if tr < (hmin * 0.1):
                Xn3 = h
                Xn = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
                Xf[0], Xf[1], Xf[2], Xf[3], Xf[4], Xf[5], Xf[6], Xf[7], Xf[8], Xf[9], Xf[10], Xf[11], Xf[12], Xf[13], Xf[14], Xf[15] = inl.v16_add(Xn, inl.sv16_mul(tr, k7))
              
                '''break the loop'''
                j = j + 1.
                
            else:
                k1 = k7                    
                X = (Xn0, Xn1, Xn2, Xn3, Xn4, Xn5, Xn6, Xn7, Xn8, Xn9, Xn10, Xn11, Xn12, Xn13, Xn14, Xn15)
    
    return


'''
 Subdivide Count: is used as part of the refinement scheme, and is designed
to consider the count value of each point: given the angle, length, ds weights
(how much we care about each value precision), do we need to insert a child point
or leave it alone or remove a point.  THIS ONLY UPDATES COUNT'''
#vec_kw = dict(nopython=True, target=TARGET)
@numba.guvectorize([replace_types( "f?[:,:], u?[:], u?, u?[:],   f?,    f?,    f?,        f?,              f? ")], "(np, nd), (d),(),(c),(),(),(),(),()", nopython=True, target='parallel')
#def subdivide_count(                  X,     depth, i,  counts, aw=1., sw=1., lw=1., thresh_remove=0.2, ds_lim=1.):
def subdivide_count(                  X,    depth,  i, counts,   aw,   sw,     lw,  thresh_remove,       ds_lim):
    werr = (aw, aw, sw, lw)
    c = 1
    
    if i < (len(X) - 1):
        eps = inl.error_aasl(X, i)
        err = inl.mag4(inl.v4_mul(eps, werr))
        
        if err > 1 and eps[3] > ds_lim:
            c = 2
            
        elif i > 0:
            if (depth[i] > depth[i-1]) and (depth[i] > depth[i+1]):
                err = err + inl.mag4(inl.v4_mul(inl.error_aasl(X, i-1), werr))
                
                if (err < thresh_remove):
                    c = 0
                    
    counts[i] = c
   # countsf[i] = c
    
    return




@numba.guvectorize([replace_types( "f?[:,:], u?, f?, f?, f?, f?[:]")], "(np, nd),(),(),(),() -> ()", nopython=True, target='parallel')
def error(X, i, aw, sw, lw, b):
    weights = (aw, aw, sw, lw)
    if i < (len(X) - 1):
        exp = inl.error_aasl(X, i)
        b[0] = inl.mag4(inl.v4_mul(exp, weights))
    return


@numba.guvectorize([replace_types( "f?[:], f?[:,:], f?[:]")], "(sixteen), (np, nd) -> (sixteen)", nopython=True, target='parallel')
def tensor_lines_velocity_call(coord, vortex, out):
    out[:] =   real(inl.calculate_lines_U_T(coord, vortex))


@numba.guvectorize([replace_types( "f?[:], f?[:,:], f?[:]")], "(sixteen), (np, nd) -> (sixteen)", nopython=True, target='parallel')
def tensor_velocity_call(coord, vortex, out):
    out[:] =   real(inl.calculate_vortex_U_T(coord, vortex))


@numba.guvectorize([replace_types( "f?[:,:], f?[:]")], "(np, nd) -> (np)", nopython=True, target='parallel')
def eigenvalues3T(A, eigenvalues):    
    
    p1 = A[0,1]**2 + A[0,2]**2 + A[1,2]**2
    
    '''might need this to account for numerical error when using small doubles in A, not just easy ints converted'''
    if p1 == real(0):
        eigenvalues[0] = A[0,0]
        eigenvalues[1] = A[1,1]
        eigenvalues[2] = A[2,2]
        
    else:
        q = real((A[0,0] + A[1,1] + A[2,2]) / real(3))
        p2 = (A[0,0] - q)**2 + (A[1,1] - q)**2 + (A[2,2] - q)**2 + real(2)*p1
        
        p = math.sqrt(p2/real(6))
        
        r = (real(2)*(A[0,1]*A[0,2]*A[1,2]) + (-A[1,2]**2 + (A[1,1]-q)*(A[2,2]-q))*(A[0,0]- q) + A[0,2]**2 *(-A[1,1]+q) + A[0,1]**2 *(-A[2,2] + q)) / (real(2)*p**3)

        if r <= real(-1):
            phi = math.pi/real(3)
            
        elif r >= real(1):
            phi = real(0)
            
        else:
            phi = math.acos(r)/real(3)
           
        eigenvalues[0] = q + real(2)*p*math.cos(phi)
        eigenvalues[2] = q + real(2)*p*math.cos(phi + (real(2)*math.pi/real(3)))
        eigenvalues[1] = real(3)*q - eigenvalues[0] - eigenvalues[2]
    return



@numba.guvectorize([replace_types( "f?[:], f?[:,:], f?, f?[:]")], "(three), (np, nd), () -> ()", nopython=True, target='parallel')
def manicottiM(coord, R, manicottiR, out):
    for i in range(len(R)): 
        di = inl.d_func(coord, R[i])
        if di < manicottiR:
            out[0]= real(0)
    return

@numba.guvectorize([replace_types( "f?[:], f?[:,:], f?[:]")], "(three), (np, nd) -> ()", nopython=True, target='parallel')
def vortexD(coord, R, out):
    maxD = real(100)
    for i in range(len(R)):
        curD = inl.d_func(coord, R[i])
        if curD < maxD:
            maxD = curD
            
    out[0] = maxD
    return


@numba.guvectorize([replace_types( "f?[:], f?[:]")], "(three) -> ()", nopython=True, target='parallel')
def maxval(A, maxvalue):    
    maxvalue[0] = max(A)
    return


@numba.guvectorize([replace_types( "f?, f?[:]")], "() -> ()", nopython=True, target='parallel')
def eval_scale(a, scale_value):    
    if a > real(2):
        scale_value[0] = real(2)
    else:
        scale_value[0] = a
    return

'''Working subdivide_resample'''
@cuda.jit
def subdivide_resample(X, di, counts, ie, Xf, df):
    i = cuda.grid(1)
    if i < len(X):
        c = counts[i]
        j = ie[i] - c
        i1 = i + 1
       
        if c > 0:
            X0 = X[i]
            d0 = di[i]

            if i < len(X)-1:
                if counts[i1]==0:
                    X0[7] += X[i1][7]
                    i1 += 1
                
            if c > 1:
                X1 = X[i1]
                d1 = di[i1]
            
                Xh0, Xh1, Xh2 = inl.v3_add(inl.sv3_mul(real(0.5), inl.v3_add(inl.xyz(X0), inl.xyz(X1)))  ,  inl.sv3_mul((X0[7] / real(8)), inl.v3_sub(inl.nxyz(X0), inl.nxyz(X1))))            
                Xh3 = real(0.5)*(X0[3] + X1[3])
                Xh4, Xh5, Xh6 = inl.v3_sub(inl.sv3_mul((real(1.5)/X0[7]), inl.v3_sub(inl.xyz(X1), inl.xyz(X0))),  inl.sv3_mul(real(0.25), inl.v3_add(inl.nxyz(X0), inl.nxyz(X1))))
                X0[7] = real(0.5)*X0[7]
                Xh7 = X0[7]
                
                Xf[j+1] = (Xh0, Xh1, Xh2, Xh3, Xh4, Xh5, Xh6, Xh7)
                df[j+1] = max(d0, d1) + 1
                
            Xf[j] = (X0[0], X0[1], X0[2], X0[3], X0[4], X0[5], X0[6], X0[7])
            
            df[j] = d0
    
    #Xf[0] = real(9), real(9), real(9), real(9), real(9), real(9), real(9), real(9)
    return


@numba.guvectorize([replace_types( "f?[:], f?[:]")], "(three) -> ()", nopython=True, target='parallel')
def get_phi(X, phi):    
    phi[0] = inl.get_phi(X)
    return

# @cuda.jit
# def subdivide_resample_retry(X, di, counts, ie, Xf, df):
#     i = cuda.grid(1)
#     if i < len(X):
#         c = counts[i]
#     return  


