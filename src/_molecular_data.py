import numpy as np
from functools import reduce
from src import Basis, examples

class MolecularData():

    def __init__(self, geometry=None, basis="sto-3g", multiplicity=1, charge=0, bohr=False, path=None, run=True):
        
        self.run = run 
        
        if run:
            a2bohr = 1.889725989
            if not bohr:
                geometry = [(atom[0], atom[1]*a2bohr) for atom in geometry]
            self._geometry = geometry
            self._basis = basis
            self._multiplicity = multiplicity
            self._charge = charge
            self.compute_n_electrons()
            self.basis_and_integrals()

        else:
            """
            read fchk, out, fock, and two-body integrals of Q-CHEM
            """
            from src import read_fchk_file, read_out_file, read_mo_ene_file,\
                            read_fock_operator, read_two_body_operator
            if path == None:
                """
                default examples: H2/STO-3G, H2/6-31G, H4/STO-3G, LiH/STO-3G, and H2O/STO-3G
                """
                # path = examples.__path__._path[0]+'/H2/sto-3g/1.60/'
                # path = examples.__path__._path[0]+'/H2/6-31g/1.6/'
                path = examples.__path__._path[0]+'/LiH/1.6/'
                # path = examples.__path__._path[0]+'/H2O/0.5/'
                # path = examples.__path__._path[0]+'/H2O/1.6/'

            with open(path+'test_qis.inp.fchk','r') as fchk:
                self.basis, self.n_basis, self.charge,\
                self.multiplicity, self.n_electrons, self.geometry = read_fchk_file(fchk)
                # self.Hcore, self.canonical_orbitals = read_fchk_file(fchk)

            with open(path+'test_qis.inp.out','r') as out:
                self.nuclear_repulsion = read_out_file(out)

            with open(path+'mo_ene_for_qis.dat','r') as mo_ene:
                # Q-CHEM order: occ aaa...abbb...b vir aaa...abbb...b
                e_occ,e_vir = read_mo_ene_file(mo_ene)
                self.orbital_energies = e_occ+e_vir
                # occ
                mo_map = {k:2*v for k, v in enumerate(np.array(e_occ).argsort())}
                mo_map.update({k+len(e_occ):2*v+1 for k, v in enumerate(np.array(e_occ).argsort())})
                # vir
                mo_map.update({k+2*len(e_occ):2*(v+len(e_occ)) for k, v in enumerate(np.array(e_vir).argsort())})
                mo_map.update({k+2*len(e_occ)+len(e_vir):2*(v+len(e_occ))+1 for k, v in enumerate(np.array(e_vir).argsort())})

            with open(path+'mo_ints_for_qis.dat','r') as fock:
                self._fock_operator = read_fock_operator(fock,mo_map)

            with open(path+'two_body_int_for_qis.dat','r') as eri:
                self._two_body_integrals = read_two_body_operator(eri,mo_map)


            # E_HF = F[i,i] - 0.5 * V[i,j,i,j] + Nuclear Repulsion
            E_HF = self.nuclear_repulsion
            for i in range(self.n_electrons):
                E_HF += self.fock_operator[i,i]
                for j in range(self.n_electrons):
                    E_HF -= 0.5 * self.two_body_integrals[i,j,i,j]
            self.hf_energy = E_HF

    def compute_n_electrons(self):
        n_electrons = {'H':1, 'He':2, 'Cu':29}
        self._n_electrons = 0 
        for atom in self.geometry:
            self._n_electrons += n_electrons.get(atom[0])
        self._n_electrons -= self.charge

    def basis_and_integrals(self):
        basis_master = Basis(self.geometry,self.basis)
        basis_master.run()

        self._n_basis = basis_master.n_basis 
        self._nuclear_repulsion = basis_master.Vnn
        self._S = basis_master.S
        self._Te = basis_master.Te
        self._Ven = basis_master.Ven
        self._Vee = basis_master.Vee

    def get_one_body_integrals(self,spatial):
        """
        One electron integrals from spatial to spin orbital representation
        """
        one_body = np.zeros((self.n_orbitals,self.n_orbitals))
        for p in range(self.n_orbitals//2):
            for q in range(self.n_orbitals//2):
                # Populate 1-body coefficients. Require p and q have same spin.
                if abs(spatial[p, q]) > 10**-15:
                    one_body[2 * p, 2 * q] = spatial[p, q]
                    one_body[2 * p + 1, 2 * q + 1] = spatial[p, q]
        return one_body

    def get_two_body_integrals(self,spatial):
        """
        Two electron integrals from spatial to spin orbital representation
        <pq||rs> anti-symmetrized
        """
        two_body = np.zeros((self.n_orbitals,self.n_orbitals,self.n_orbitals,self.n_orbitals))
        for p in range(self.n_orbitals//2):
            for q in range(self.n_orbitals//2):
                for r in range(self.n_orbitals//2):
                    for s in range(self.n_orbitals//2):

                        if abs(spatial[p,q,r,s]) > 10**-13:
                            # Vee[phi_i,phi_i,phi_j,phi_j] physics? chemist? notation
                            # anyway the way that I am not familiar with
                            # I should do [pq|rs] -> <pr|qs>
                            # print(p,q,r,s, spatial[p,q,r,s])

                            two_body[2*p, 2*r+1, 2*q, 2*s+1] += spatial[p,q,r,s]
                            two_body[2*r+1, 2*p, 2*s+1, 2*q] += spatial[p,q,r,s]
                            two_body[2*q, 2*s+1, 2*p, 2*r+1] += spatial[p,q,r,s]
                            two_body[2*s+1, 2*q, 2*r+1, 2*p] += spatial[p,q,r,s]
                            two_body[2*p, 2*r+1, 2*s+1, 2*q] -= spatial[p,q,r,s]
                            two_body[2*r+1, 2*p, 2*q, 2*s+1] -= spatial[p,q,r,s]
                            two_body[2*s+1, 2*q, 2*p, 2*r+1] -= spatial[p,q,r,s]
                            two_body[2*q, 2*s+1, 2*r+1, 2*p] -= spatial[p,q,r,s]
                            
                            two_body[2*p+1, 2*r, 2*q+1, 2*s] += spatial[p,q,r,s]
                            two_body[2*r, 2*p+1, 2*s, 2*q+1] += spatial[p,q,r,s]
                            two_body[2*q+1, 2*s, 2*p+1, 2*r] += spatial[p,q,r,s]
                            two_body[2*s, 2*q+1, 2*r, 2*p+1] += spatial[p,q,r,s]
                            two_body[2*p+1, 2*r, 2*s, 2*q+1] -= spatial[p,q,r,s]
                            two_body[2*r, 2*p+1, 2*q+1, 2*s] -= spatial[p,q,r,s]
                            two_body[2*s, 2*q+1, 2*p+1, 2*r] -= spatial[p,q,r,s]
                            two_body[2*q+1, 2*s, 2*r, 2*p+1] -= spatial[p,q,r,s]

                            if p != r and q != s:
                                two_body[2*p, 2*r, 2*q, 2*s] += spatial[p,q,r,s]
                                two_body[2*r, 2*p, 2*s, 2*q] += spatial[p,q,r,s]
                                two_body[2*q, 2*s, 2*p, 2*r] += spatial[p,q,r,s]
                                two_body[2*s, 2*q, 2*r, 2*p] += spatial[p,q,r,s]
                                two_body[2*p, 2*r, 2*s, 2*q] -= spatial[p,q,r,s]
                                two_body[2*r, 2*p, 2*q, 2*s] -= spatial[p,q,r,s]
                                two_body[2*s, 2*q, 2*p, 2*r] -= spatial[p,q,r,s]
                                two_body[2*q, 2*s, 2*r, 2*p] -= spatial[p,q,r,s]
                                
                                two_body[2*p+1, 2*r+1, 2*q+1, 2*s+1] += spatial[p,q,r,s]
                                two_body[2*r+1, 2*p+1, 2*s+1, 2*q+1] += spatial[p,q,r,s]
                                two_body[2*q+1, 2*s+1, 2*p+1, 2*r+1] += spatial[p,q,r,s]
                                two_body[2*s+1, 2*q+1, 2*r+1, 2*p+1] += spatial[p,q,r,s]
                                two_body[2*p+1, 2*r+1, 2*s+1, 2*q+1] -= spatial[p,q,r,s]
                                two_body[2*r+1, 2*p+1, 2*q+1, 2*s+1] -= spatial[p,q,r,s]
                                two_body[2*s+1, 2*q+1, 2*p+1, 2*r+1] -= spatial[p,q,r,s]
                                two_body[2*q+1, 2*s+1, 2*r+1, 2*p+1] -= spatial[p,q,r,s]
        return 0.25 * two_body

    @property
    def geometry(self):
        """ molecular coodinates """
        return self._geometry
    @geometry.setter
    def geometry(self, xyz):
        self._geometry = xyz 

    @property
    def basis(self):
        """ name of basis set """
        return self._basis.lower()
    @basis.setter
    def basis(self, basis):
        self._basis = basis
   
    @property
    def multiplicity(self):
        """ multiplicity """
        return self._multiplicity
    @multiplicity.setter
    def multiplicity(self, multiplicity):
        self._multiplicity = multiplicity
    
    @property
    def charge(self):
        """ charge """
        return self._charge
    @charge.setter
    def charge(self, charge):
        self._charge = charge

    @property
    def n_electrons(self):
        """ number of electrons """
        return self._n_electrons
    @n_electrons.setter
    def n_electrons(self, value):
        self._n_electrons = value

    @property
    def nuclear_repulsion(self):
        """ nuclear_repulsion energy """
        return self._nuclear_repulsion
    @nuclear_repulsion.setter
    def nuclear_repulsion(self, value):
        self._nuclear_repulsion = value

    @property
    def n_basis(self):
        """ number of basis functions """
        return self._n_basis
    @n_basis.setter
    def n_basis(self, n_basis):
        self._n_basis = n_basis
    
    @property
    def n_orbitals(self):
        """ number of spin orbitals """
        return self.n_basis * 2
    @n_orbitals.setter
    def n_orbitals(self, n_orbitals):
        self._n_orbitals = n_orbitals
 
    @property
    def S(self):
        """ overlap matrix """
        return self._S
    @S.setter
    def S(self, value):
        self._S = value

    @property
    def Te(self):
        """ kinetic energy of electron"""
        return self._Te
    @Te.setter
    def Te(self, value):
        self._Te = value

    @property
    def Ven(self):
        """ electron-nuclear attraction """
        return self._Ven
    @Ven.setter
    def Ven(self, value):
        self._Ven = value

    @property
    def Vee(self):
        """ electron-electron repulsion """
        return self._Vee
    @Vee.setter
    def Vee(self, value):
        self._Vee = value

    @property
    def hf_energy(self):
        """ Hartree-Fock energy """
        return self._hf_energy
    @hf_energy.setter
    def hf_energy(self, hf_energy):
        self._hf_energy = hf_energy

    @property
    def mp2_energy(self):
        """ MP2 energy """
        return self._mp2_energy
    @mp2_energy.setter
    def mp2_energy(self, Ecorr):
        self._mp2_energy = Ecorr 

    @property
    def Hcore(self):
        """ Core Hamiltonian """
        return self._Hcore
    @Hcore.setter
    def Hcore(self, Hcore):
        self._Hcore = Hcore 

    @property
    def canonical_orbitals(self):
        """ Hartree-Fock canonical orbital coefficients (represented on AO basis) """
        return self._canonical_orbitals
    @canonical_orbitals.setter
    def canonical_orbitals(self, Cao):
        self._canonical_orbitals = Cao

    @property
    def orbital_energies(self):
        """ canonical orbital_energies """
        return self._orbital_energies
    @orbital_energies.setter
    def orbital_energies(self, Emo):
        self._orbital_energies = Emo
    
    @property
    def fock_operator(self):
        """ Fock_operator """
        if self.run:
            self._fock_operator = self.get_one_body_integrals(np.diag(self.orbital_energies))
        return self._fock_operator

    @property
    def one_body_integrals(self):
        """ 
        one-electron integrals (spin orbital representation)
        hpq = MO_dagger * H_core * MO
        """
        mo      = self.canonical_orbitals
        h_core  = self.Hcore
        spatial = reduce(np.dot, (mo.T, h_core, mo))
        self._one_body_integrals = self.get_one_body_integrals(spatial)
        return self._one_body_integrals

    @property
    def two_body_integrals(self):
        """
        two-body integrals (spin orbital representation)
        Vpqrs = sum_mu,nu,sigma,gamma (C_mu^p C_nu^q C_sigma^r C_gamma^s * V_mu,nu,sigma,gamma (AO basis eri))
        """
        if self.run:
            spatial = np.zeros((self.n_basis,self.n_basis,self.n_basis,self.n_basis))
            for p in range(self.n_basis):
                for q in range(self.n_basis):
                    for r in range(self.n_basis):
                        for s in range(self.n_basis):
                            for mu in range(self.n_basis):
                                for nu in range(self.n_basis):
                                    for sigma in range(self.n_basis):
                                        for gamma in range(self.n_basis):
                                            spatial[p,q,r,s] += self.canonical_orbitals[mu,p] * \
                                                                self.canonical_orbitals[nu,q] * \
                                                                self.canonical_orbitals[sigma,r] * \
                                                                self.canonical_orbitals[gamma,s] * \
                                                                self.Vee[mu,nu,sigma,gamma]

            self._two_body_integrals = self.get_two_body_integrals(spatial)
        return self._two_body_integrals

    @property
    def mp2_energy(self):
        """ mp2_energy """
        return self._mp2_energy
    @mp2_energy.setter
    def mp2_energy(self, Ecorr):
        self._mp2_energy = Ecorr 

    @property
    def ccsd_energy(self):
        """ ccsd_energy """
        return self._ccsd_energy
    @ccsd_energy.setter
    def ccsd_energy(self, Ecorr):
        self._ccsd_energy = Ecorr

if __name__ == "__main__":

    molecule1 = MolecularData(run=False)
    
    basis = '6-31g'
    geometry = [('H', np.array([0., 0., 0.])), ('H', np.array([0., 0., 1.8]))]
    multiplicity = 1
    charge = 0
    molecule2 = MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
    print(molecule2.Vee)

