import numpy as np
from src import MolecularData

class MBPT():

    """
    Moller-Plasset Perturbation Theroy
    kth order correction:
     - E_{corr}^{k} = <Phi_{0} | V (RV)^{k-1} | Phi_{0}>
     - |Psi^{k}> = (RV)^{k} |Phi_{0}>
    """

    def __init__(self,molecule):
        self.molecule = molecule
        self.n_occ = molecule.n_electrons
        self.n_vir = molecule.n_orbitals - molecule.n_electrons
        o,v = self.n_occ,self.n_vir
        self.f_oo = molecule.fock_operator[:o, :o]
        self.f_vv = molecule.fock_operator[o:o+v, o:o+v]
        self.i_oovv = molecule.two_body_integrals[:o, :o, o:o+v, o:o+v]

    def run(self,order=2):

        if order == 2:
            self.run_mp2()
    
    def run_mp2(self):

        mp2_corr = 0.0
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for a in range(self.n_vir):
                    for b in range(self.n_vir):
                        Dijab = self.f_oo[i,i] + self.f_oo[j,j] - self.f_vv[a,a] - self.f_vv[b,b]
                        mp2_corr += 0.25 * self.i_oovv[i,j,a,b]**2 / Dijab 
        
        print("\nTotal  MP2   correlation     energy = {:.10f} au".format(mp2_corr))
        print("       MP2         total     energy = {:.10f} au".format(mp2_corr+self.molecule.hf_energy))

        self.molecule.mp2_energy = mp2_corr


if __name__ == "__main__":
    pass
