


fgsm_array = [21.19, 21.46, 23.200000000000003, 30.15, 57.4, 77.39, 86.97, 90.68, 92.28, 93.26, 92.74, 92.86999999999999, 
              91.51, 92.22, 91.74, 91.36999999999999, 89.86, 90.0, 89.21, 90.14999999999999]

pgd_array = [1.13, 1.1199999999999999, 1.08, 0.98, 16.98, 55.61000000000001, 80.2, 90.64999999999999, 94.62, 96.17999999999999, 97.11999999999999, 
             97.44, 97.5, 97.63, 97.64, 97.84, 97.81, 97.72999999999999, 97.8, 97.63]


import matplotlib.pyplot as plt
import numpy as np

max_eigen_value_array = [-0.8395181894302368, -1.9403785467147827, -2.970109224319458, -3.8656251430511475, -4.628428936004639, 
                         -5.303205966949463, -5.901797771453857, -6.435247421264648, -6.914281845092773, -7.348176956176758, 
                         -7.744332313537598, -8.108587265014648, -8.445579528808594, -8.759049415588379, -9.052019119262695, 
                         -9.326985359191895, -9.586030960083008, -9.830892562866211, -10.063036918640137, -10.283724784851074]

max_eigen_value_array = np.array(max_eigen_value_array)*-0.001
plt.figure
plt.plot(max_eigen_value_array,fgsm_array)
plt.plot(max_eigen_value_array,pgd_array)
plt.xlabel(r'Contraction rate $\rho$')
plt.ylabel("Robust Accuracy with $epsilon$ = 0.3 (%)")
plt.legend(['FGSM','PGD'])
plt.show()
plt.savefig("/home/mzakwan/neurips2023/MNIST/fgsmVsEigenvalue.pdf")