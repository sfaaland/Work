#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import mesh
π = np.pi

p = 2
q = 3
N = 200
N_tube = 20

R = 1
a = 0.25

ϕ = np.linspace(0, 2*π, N, False)

sq = np.sin(q * ϕ)
cq = np.cos(q * ϕ)
sp = np.sin(p * ϕ)
cp = np.cos(p * ϕ)

r = R + a * cq
z = a * sq

X = np.empty((N, 3))
X[:, 0] = r * cp
X[:, 1] = r * sp
X[:, 2] = z

# We're going to create an orthonormal basis about each point.
B = np.zeros((N, 2, 3))
# The first attached vector is the tangent, unormalized
B[:, 0, 0] = -p * r * sp - a * q * cp * sq
B[:, 0, 1] =  p * r * cp - a * q * sp * sq
B[:, 0, 2] = a * q * cq
# The second vector is just z-hat -- this will get turned into an orthogonal
#   vector to the first when we normalize the basis!
B[:, 1, 2] = 1

B = mesh.normalize_basis(B)

m = mesh.extrude_shape(X, B[:, 1], B[:, 2], outline=N_tube, scale=0.15 * (1.5 + np.sin(9 * ϕ)))

# We need ϕ for each point in the mesh, but there are N_tube vertices per point
mϕ = np.empty((N, N_tube))
mϕ[:, :] = ϕ.reshape(N, 1)
m.colorize(mϕ.flat, cmap='hsv')

m.save('test.ply')
