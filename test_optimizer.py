import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft

# #################
# Import optimizer
# #################
from optimizers.gpsr_basic import GPSR_Basic
# ##################

# Create an example for signal reconstruction
n = 5000
t = np.linspace(0, 1/8, n)
y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
yt = spfft.dct(y, norm='ortho')

# extract small sample of signal
m = 500
ri = np.random.choice(n, m, replace=False)
ri.sort()
t2 = t[ri]
y2 = y[ri]

A = spfft.idct(np.identity(n), norm='ortho', axis=0)
A = A[ri]

y2 = y2.reshape(-1, 1)
yt = yt.reshape(-1, 1)


# ###############################
# Test the optimizer === x, _ = optimizer(y, A, ...)
# ###############################

x, _ = GPSR_Basic(y=y2, A=A, tau=np.array([0.08]))

# ###############################

x = np.squeeze(x)
sig = spfft.idct(x, norm='ortho', axis=0)

plt.figure()
plt.subplot(221)
plt.plot(yt)
plt.title("Original DCT (Sparse signal)")
plt.subplot(222)
plt.plot(x)
plt.title("Result from the optimizer")
plt.subplot(223)
plt.plot(y[2000:3000])
plt.title("Original Signal")
plt.subplot(224)
plt.plot(sig[2000:3000])
plt.title("Reconstructed signal")
plt.show()

