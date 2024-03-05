import numpy as np
from svc import SupportVectorMachine
import matplotlib.pylab as plt

X = np.concatenate((np.random.randn(500, 2) - 2, np.random.randn(500, 2) + 2))
y = np.concatenate((np.ones(500), -np.ones(500)))
C = SupportVectorMachine(iteration=100)
C.fit(X, y)
w, b = C.weight
u = np.linspace(-3, 3, 100)
v = (-b - w[0] * u) / w[1]

plt.scatter(X[:500, 0], X[:500, 1], label='Positive')
plt.scatter(X[500:, 0], X[500:, 1], label='Negative')
plt.plot(u, v, label='Separation', c='g')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Separation Sample')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./figure/separation.png')
plt.show()