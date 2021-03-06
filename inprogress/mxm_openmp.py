# coding: utf-8

from numpy import zeros

#$ header mxm_omp(int, int, int, double[:,:], double[:,:], double[:,:])
def mxm_omp(n,m,p,a,b,c):
    a = 0.
    b = 0.
    c = 0.

    #$ omp parallel
    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = i-j
    #$ omp end do nowait

    #$ omp do schedule(runtime)
    for i in range(0, m):
        for j in range(0, p):
            b[i,j] = i+j
    #$ omp end do nowait

    #$ omp do schedule(runtime)
    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, p):
                c[i,j] = c[i,j] + a[i,k]*b[k,j]
    #$ omp end do
    #$ omp end parallel


n = 800
m = 1600
p = 800

a = zeros((n,m))
b = zeros((m,p))
c = zeros((n,p))

mxm_omp(n,m,p,a,b,c)
