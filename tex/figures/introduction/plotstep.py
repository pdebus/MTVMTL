import matplotlib.pyplot as plt
import numpy as np


x1=np.linspace(0,1,11)
x2=np.linspace(0,1,2)
x3=np.linspace(0,1,101)

plt.ylim(-0.01, 1.01)
plt.plot(x3, x3, linestyle='-', linewidth=3, label=r'$f_3(x)$')
plt.step(x1, x1, linestyle='--', linewidth=3, drawstyle='steps-mid', label=r'$f_2(x)$')
plt.step(x2, x2, linestyle='-.', linewidth=3, drawstyle='steps-mid', label=r'$f_1(x)$')
plt.legend(loc='upper left')
plt.savefig("tv12comparison.pdf")
plt.savefig("tv12comparison.png")
plt.show()
