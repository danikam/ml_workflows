"""
  Date:     200314
  Purpose:  Convert dataset from sklearn into txt format
"""

from sklearn.datasets import make_moons, make_circles, make_classification

X, classification = make_moons(noise=0.3, random_state=0)

x = X[:,0]
y = X[:,1]

f=open("moon.txt", 'w')
f.write("%12s%12s%12s\n"%("X", "Y", "Value"))
for i in range(len(x)):
  f.write("%12f%12f%12i\n"%(x[i], y[i], classification[i]))
f.close()
