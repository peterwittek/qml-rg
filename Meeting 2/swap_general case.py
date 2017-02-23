from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# First state
alpha1=np.asscalar(np.random.rand(1,1))
beta1=np.sqrt(1-alpha1**2)
ket1 = alpha1*basis(2, 0) + beta1*basis(2, 1)


# Second state
alpha2=np.asscalar(np.random.rand(1,1))
beta2=np.sqrt(1-alpha2**2)
ket2 = alpha2*basis(2, 0) + beta2*basis(2, 1)


# Ancilla
anc=(basis(2, 0) + basis(2, 1)).unit()

# Input
psi_in=tensor(anc,ket1,ket2)

# Swap gate = fredkin+hadamar
output=-hadamard_transform(N=3)*fredkin()*tensor(anc,ket1,ket2) # the minus is needed for hadarmad_transform(N=3) to work as 
                                                                # hadamard gate
                        
# Measurement
down=basis(2,0)
up=basis(2,1)

result_down=tensor(down,ket1,ket2).dag()*output
result_up=tensor(up,ket1,ket2).dag()*output
                
# Checking results
# With some easy algebra we can calculate the theoretical result for both measurements.
# In this case, I have calculated the expected value for the state up=|1>

expected_up=((beta1*alpha2)**2+(beta2*alpha1)**2-2*alpha1*beta1*alpha2*beta2)*(1/2)

print(result_up)
print(expected_up)
