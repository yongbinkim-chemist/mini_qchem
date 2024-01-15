import numpy as np
from src import MolecularData

class SCF():
    """
    Hartree-Fock calculation
    """
    def __init__(self,molecule):
        self._molecule = molecule
        self._n_basis = molecule.n_basis
        self._n_electrons = molecule.n_electrons

    def run(self,n_iter=50,tol=1.0e-6):
        """
        Solve FC = eSC due to our basis set is not orthonormal (S != I)
        F = Hcore + G = Te + Ven + G, where G = 2J-K, two-electron integrals

        1. SL = DL (diagonalize S)
        2. build operator S^{-1/2} = U = L D^{-1/2} L^{\dagger}
            - U^{\dagger} S U = U S U^{\dagger} = I
            - U will transform F from AO to MO basis
        3. Solve
            Fmo = U^{\dagger} Fao U
            FmoCmo = eCmo -> Cmo: cols->MOs rows->AOs
            Cao = U Cmo
        """

        electronic_energy = 0.0

        # it is exact if no e-e repulsion
        Hcore = self._molecule.Te + self._molecule.Ven
        density_matrix = np.zeros((self._n_basis,self._n_basis))
        
        D, L = np.linalg.eig(self._molecule.S)
        # S_eval_minhalf = np.diag(D**(-0.5))
        U = np.dot(L, np.dot(np.diag(D**(-0.5)), L.transpose().conj()))

        conv = False
        print(" --------------------------------------- ")
        print("  Cycle       Energy       Delta Energy  ")
        print(" --------------------------------------- ")
        for iter in range(n_iter):

            F = self.build_fock_matrix(Hcore, density_matrix) # in AO basis

            Fmo = np.dot(U.transpose().conj(), np.dot(F, U))
            # Emo,Cmo = np.linalg.eig(Fmo) # E = Molecular Orbital Energy and C = Molecular Orbital
            """
            eigh: works for symmetric matrix
                  if not symmetric, use eig -> then energies might not be ordered
            """
            Emo,Cmo = np.linalg.eigh(Fmo) # E = Molecular Orbital Energy and C = Molecular Orbital
            Cao = np.dot(U,Cmo)

            # print(Emo,'\n', Cao)
            density_matrix = self.compute_density_matrix(Cao)
            old_energy = electronic_energy
            electronic_energy = self.compute_energy(Hcore,F,density_matrix)
            delta_energy = abs(old_energy-electronic_energy)

            if delta_energy < tol:
                conv = True
                self._molecule.hf_energy = electronic_energy + self._molecule.nuclear_repulsion
                self._molecule.Hcore = Hcore
                self._molecule.canonical_orbitals = Cao
                self._molecule.orbital_energies = Emo
                print("{0:5d} {1:17.10f}* {2:14.10f}".format(iter, electronic_energy+self._molecule.nuclear_repulsion, delta_energy))
                print(" --------------------------------------- ")
                print("SCF   energy in the final basis set = {:.10f}".format(self._molecule.hf_energy))
                print("Total energy in the final basis set = {:.10f}".format(self._molecule.hf_energy))
                break

            # { n : 0 k . j f}".format( p1, p2, ... )
            # nth element, fill 0, k space, j decimal, f float
            # integer case {n: 0 k d}, there should not be j
            print("{0:5d} {1:17.10f} {2:15.10f}".format(iter, electronic_energy+self._molecule.nuclear_repulsion, delta_energy))
        
        if not conv:
           raise Exception("SCF Failed -- Iterations exceeded")

    def build_fock_matrix(self, Hcore, density_matrix):
        ####################################################################################################################
        # F_mu,nu = Hcore_mu,nu + <Phi_mu|2J_j - K_j|Phi_nu>                                                               #
        #        = Hcore_mu,nu + \sum_{sigma,gamma} C_sigma^{j}C_gamma{j} (<Phi_mu Phi_sigma|(1/r)|Phi_nu Phi_gamma>)      #
        #        = Hcore_mu,nu + \sum_{sigma,gamma} D_sigma,gamma * (Vee[mu,nu,sigma,gamma] - 0.5*Vee[mu,sigma,nu,gamma])) #
        ####################################################################################################################
        F = np.zeros((self._n_basis,self._n_basis))
        for mu in range(self._n_basis):
            for nu in range(self._n_basis):
                F[mu,nu] += Hcore[mu,nu]
                for sigma in range(self._n_basis):
                    for gamma in range(self._n_basis):
                        # Vee[phi_i,phi_i,phi_j,phi_j]
                        F[mu,nu] += density_matrix[sigma,gamma] * (self._molecule.Vee[mu,nu,sigma,gamma] - 0.5*self._molecule.Vee[mu,gamma,sigma,nu])
        return F

    def compute_density_matrix(self, Cao):
        ######################################################
        # Dmu,nu = \sum_{i}^{nocc} C_mu^i^{\dagger} * C_nu^i #
        ######################################################
        density_matrix = np.zeros((self._n_basis,self._n_basis))
        occupation = 2.0
        for mu in range(self._n_basis):
            for nu in range(self._n_basis):
                for oo in range(self._n_electrons//2):
                    density_matrix[mu,nu] += occupation * Cao[mu,oo] * Cao[nu,oo]
        return density_matrix

    def compute_energy(self, Hcore, F, density_matrix):
        ##############################################################################################################
        # E_HF = \sum_{i} <psi_i|Hcore + 0.5 * \sum_{j}(J_j - K_j)|psi_i>                                            #
        #      = \sum_{i} \sum_{pq} Cpq (Hcore_pq + 0.5*\sum_{rs} D_rs*(0.5*Vee[p,r,q,s]-Vee[p,r,s,q]))              #
        #      = \sum_{i} \sum_{pq} 0.5 * Cpq (2Hcore_pq + \sum_{rs} D_rs*(0.5*Vee[p,r,q,s]-Vee[p,r,s,q]))           #
        #      = \sum_{i} \sum_{pq} 0.5 * Cpq (Hcore_pq + Hcore_pq + \sum_{rs} D_rs*(0.5*Vee[p,r,q,s]-Vee[p,r,s,q])) #
        #      = \sum_{i} \sum_{pq} 0.5 * Cpq (Hcore_pq + F_pq)                                                      #
        ##############################################################################################################
        E_HF = 0.0E0  
        for mu in range(self._n_basis):  
            for nu in range(self._n_basis):  
                E_HF += 0.5 * density_matrix[mu,nu] * (Hcore[mu,nu] + F[mu,nu])
        return E_HF

if __name__ == "__main__":
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    # geometry = [('He', np.array([0., 0., 0.])), ('H', np.array([0.74, 0., 0.]))]
    geometry = [('H', np.array([0., 0., 0.])), ('H', np.array([1.4, 0., 0.]))]
    molecule = MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
    run_scf = SCF(molecule)
    run_scf.run()
