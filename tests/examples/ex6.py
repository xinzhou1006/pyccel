# coding: utf-8

#from numpy import zeros
#from numpy import linspace

a = zeros(shape=64, dtype=float)
b = zeros(shape=8, dtype=int)

a[1] = 1.0
a[2] = 1.0
a[3] = 1.0

c = a[1]

d = c + 5.3 * a[1+1] + 4.0 - a[3]
print(d)

#e = a # not working. e must be declared as an array

x = zeros(shape=(2,8), dtype=float)
x[1,1] = 1

y = x[0,2]
print(y)

#z = linspace(0,4,1)
#t = z[1:3]
