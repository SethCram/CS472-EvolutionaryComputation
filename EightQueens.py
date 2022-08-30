import numpy
import matplotlib.pyplot as plt

"""
#plot1:
plt.figure(figsize = (10,15)) #1st arg = horizontal space, 2nd arg = vertical space
plt.subplot(3, 1, 1)
plt.plot(t, f1) # t = step size, smaller it is the more accurate graph is
                # y = array of cos vals at each step
plt.grid() #add a grid to graph
plt.title('f1(t) vs t')
plt.ylabel('f1(t)')
"""

steps = 1e-2 # Define step size
t = numpy.arange(0, 20 + steps , steps)

data = (3, 6, 9, 12)

fig, simple_chart = plt.subplots()

simple_chart.plot(data)

plt.show()