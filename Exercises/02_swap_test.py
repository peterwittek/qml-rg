import qutip as qt
import numpy as np

"""
	            fredkin	hadamard
ancilla---------O---------H------------Measure
     f1---------X----------------------
     f2---------X----------------------
"""

up = qt.qstate('u')					#qstate down is (1,0), qstate up is (0,1)
down = qt.qstate('d')
ancilla = 1/np.sqrt(2)*(up + down) 			
hadamard = qt.tensor(qt.snot(),qt.qeye(2),qt.qeye(2))
fredkin = qt.fredkin()					#fredkin swapes f1 and f2 if ancilla = 1

#------------------------------------------------------------------------
#Define Input
f1 = 1/np.sqrt(2)*(up + down)
f2 = up
#------------------------------------------------------------------------

psi0 = qt.tensor(ancilla, f1, f2)

output = hadamard*fredkin*psi0


measure = qt.tensor(down,f1,f2).dag()*output 	#if f1 = f2 the output is (down,f1,f2)
					  	#therefore measure is 1 if f1 = f2 
                        #measure is 0.5 if f1, f2 are orthogonal
						
print(measure)
