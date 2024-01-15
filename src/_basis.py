import numpy as np
from scipy import special

class Basis():

    """
    Gaussian Basis
    Phi(x,y,z) = Aexp(-a[(x-x0)^2 + (y-y0)^2 + (z-z0)^2]) = Aexp(-a|r-r0|^2)
    A = (2a/pi)^(3/4), wavefunction should be normalized
    "a" will be given, (contraction coefficients and exponents), x0,y0,z0 -> nuclear coordinates
    """

    def __init__(self,geometry,basis):
        self._geometry = geometry
        self._basis = basis

    def run(self):
        self._basis_function = []
        for atom in self._geometry:
            self._basis_function.extend(self.get_basis_function(atom))
        self._n_basis = len(self._basis_function)
        
        self.compute_nuclear_nuclear_repulsion()
        self.compute_overlap_integral()
        self.compute_kinetic_energy_of_electron()
        self.compute_electron_nuclear_attraction()
        self.compute_electron_electron_repulsion()

    def get_basis_function(self,atom):
        proton, xyz = atom

        if proton == "Cu":
            return [STO2G(proton, xyz)]

        if proton == "H":
            if self._basis == "sto-3g":
                return [STO3G(proton, xyz)]
            elif self._basis == "6-31g":
                return [Six31G(proton, xyz, 0),Six31G(proton, xyz, 1)]

        if proton == "He":
            if self._basis == "sto-3g":
                return [STO3G(proton, xyz)]
            elif self._basis == "6-31g":
                return [Six31G(proton, xyz, 0),Six31G(proton, xyz, 1)]
        """
        Add more
        """
    
    def boys(self, x, n):
        if x == 0:
            return 1.0/(2*n+1)
        else:
            return special.gammainc(n+0.5,x) * special.gamma(n+0.5) * (1.0/(2*x**(n+0.5)))

    def compute_overlap_integral(self):
        """
        integral Phi_1(x,y,z) Phi_2(x,y,z) dxdydz
        """
        S = np.zeros((self.n_basis,self.n_basis))
        for i, func_i in enumerate(self._basis_function):
            for j, func_j in enumerate(self._basis_function):

                nprimitive_i = func_i.nprimitive
                nprimitive_j = func_j.nprimitive 

                for ii in range(nprimitive_i):
                    for jj in range(nprimitive_j):
                        norm = func_i.norm[ii] * func_j.norm[jj]

                        """gaussian product theorem"""
                        p = func_i.alpha[ii] + func_j.alpha[jj]
                        q = func_i.alpha[ii] * func_j.alpha[jj] / p
                        Q = func_i.xyz - func_j.xyz
                        Q2 = np.dot(Q,Q)
                        S[i,j] += norm * func_i.coeff[ii] * func_j.coeff[jj] * np.exp(-q*Q2) * (np.pi/p)**(3/2)
        self._S = S

    def compute_kinetic_energy_of_electron(self):
        """
        integral Phi_1(x,y,z) (-1/2) (d^2/dx^2+d^2/dy^2+d^2/dz^2) Phi_2(x,y,z)
        """
        Te = np.zeros((self.n_basis,self.n_basis))
        for i, func_i in enumerate(self._basis_function):
            for j, func_j in enumerate(self._basis_function):

                nprimitive_i = func_i.nprimitive
                nprimitive_j = func_j.nprimitive 

                for ii in range(nprimitive_i):
                    for jj in range(nprimitive_j):
                        norm = func_i.norm[ii] * func_j.norm[jj]

                        """ gaussian product theorem """
                        p = func_i.alpha[ii] + func_j.alpha[jj]
                        q = func_i.alpha[ii] * func_j.alpha[jj] / p
                        Q = func_i.xyz - func_j.xyz
                        Q2 = np.dot(Q,Q)

                        """ in addition to the overlap """
                        P = func_i.alpha[ii] * func_i.xyz + func_j.alpha[jj] * func_j.xyz
                        Pp = P/p
                        PG = Pp - func_j.xyz 
                        PG2 = np.square(PG)
                        # overlap element
                        s = norm * func_i.coeff[ii] * func_j.coeff[jj] * np.exp(-q*Q2) * (np.pi/p)**(3/2)
                        Te[i,j] += 3 * func_j.alpha[jj] * s 
                        Te[i,j] -= 2 * func_j.alpha[jj] * func_j.alpha[jj] * s * (PG2[0] + (1 / (2*p)))
                        Te[i,j] -= 2 * func_j.alpha[jj] * func_j.alpha[jj] * s * (PG2[1] + (1 / (2*p)))
                        Te[i,j] -= 2 * func_j.alpha[jj] * func_j.alpha[jj] * s * (PG2[2] + (1 / (2*p)))
        self._Te = Te 

    def compute_electron_nuclear_attraction(self):
        """
        integral Phi_1(x,y,z) (-ZA / |r_i - R_A|) Phi_2(x,y,z)
        """
        proton = {"H":1, "He":2, "Cu":29} # add more
        Ven = np.zeros((self.n_basis,self.n_basis))
        for atom in self._geometry:
            Z = proton.get(atom[0])
            # xyz = np.array(atom[1])
            xyz = atom[1]

            for i, func_i in enumerate(self._basis_function):
                for j, func_j in enumerate(self._basis_function):

                    nprimitive_i = func_i.nprimitive
                    nprimitive_j = func_j.nprimitive 

                    for ii in range(nprimitive_i):
                        for jj in range(nprimitive_j):
                            norm = func_i.norm[ii] * func_j.norm[jj]

                            """ gaussian product theorem """
                            p = func_i.alpha[ii] + func_j.alpha[jj]
                            q = func_i.alpha[ii] * func_j.alpha[jj] / p
                            Q = func_i.xyz - func_j.xyz
                            Q2 = np.dot(Q,Q)

                            """ in addition to the overlap """
                            P = func_i.alpha[ii] * func_i.xyz + func_j.alpha[jj] * func_j.xyz
                            Pp = P/p
                            PG = Pp - xyz
                            PG2 = np.dot(PG,PG)

                            Ven[i,j] += -Z * norm * func_i.coeff[ii] * func_j.coeff[jj] * np.exp(-q*Q2) * (2.0*np.pi/p) * self.boys(p*PG2, 0) 
        self._Ven = Ven 

    def compute_electron_electron_repulsion(self):
        """
        integral Phi_1(x,y,z) Phi_2(x,y,z) (1 / |r_i - r_j|) Phi_3(x,y,z) Phi_4(x,y,z)
        """
        Vee = np.zeros((self.n_basis,self.n_basis,self.n_basis,self.n_basis))

        for i, func_i in enumerate(self._basis_function):
            for j, func_j in enumerate(self._basis_function):
                for k, func_k in enumerate(self._basis_function):
                    for l, func_l in enumerate(self._basis_function):

                        nprimitive_i = func_i.nprimitive
                        nprimitive_j = func_j.nprimitive
                        nprimitive_k = func_k.nprimitive 
                        nprimitive_l = func_l.nprimitive 

                        for ii in range(nprimitive_i):
                            for jj in range(nprimitive_j):
                                for kk in range(nprimitive_k):
                                    for ll in range(nprimitive_l):

                                        norm = func_i.norm[ii] * func_j.norm[jj] * func_k.norm[kk] * func_l.norm[ll]
                                        cijkl = func_i.coeff[ii] * func_j.coeff[jj] * func_k.coeff[kk] * func_l.coeff[ll]

                                        pij = func_i.alpha[ii] + func_j.alpha[jj]
                                        pkl = func_k.alpha[kk] + func_l.alpha[ll]

                                        Pij = func_i.alpha[ii] * func_i.xyz + func_j.alpha[jj] * func_j.xyz
                                        Pkl = func_k.alpha[kk] * func_k.xyz + func_l.alpha[ll] * func_l.xyz

                                        Ppij = Pij/pij
                                        Ppkl = Pkl/pkl

                                        PpijPpkl = Ppij - Ppkl
                                        PpijPpkl2 = np.dot(PpijPpkl,PpijPpkl)

                                        denom = 1.0/pij + 1.0/pkl

                                        qij = func_i.alpha[ii] * func_j.alpha[jj] / pij
                                        qkl = func_k.alpha[kk] * func_l.alpha[ll] / pkl

                                        Qij = func_i.xyz - func_j.xyz
                                        Qkl = func_k.xyz - func_l.xyz

                                        Qij2 = np.dot(Qij,Qij)
                                        Qkl2 = np.dot(Qkl,Qkl)

                                        term1 = 2.0 * np.pi * np.pi / (pij * pkl)
                                        term2 = np.sqrt(np.pi / (pij + pkl))
                                        term3 = np.exp(-qij * Qij2)
                                        term4 = np.exp(-qkl * Qkl2)

                                        Vee[i,j,k,l] += norm * cijkl * term1 * term2 * term3 * term4 * self.boys(PpijPpkl2/denom,0)
        self._Vee = Vee 

    def compute_nuclear_nuclear_repulsion(self):
        """
        sum_{A,B} ZA*ZB / |RA - RB|
        """
        Vnn = 0.0 
        proton = {"H":1, "He":2, "Cu":29} # add more

        for i in range(len(self._geometry)):
            ZA = proton.get(self._geometry[i][0])
            # RA = np.array(self._geometry[i][1])
            RA = self._geometry[i][1]

            for j in range(i+1, len(self._geometry)):
                ZB = proton.get(self._geometry[j][0])
                RB = self._geometry[j][1]

                RAB = np.sqrt(np.sum(np.square(RA-RB)))
                Vnn += ZA * ZB / RAB
                
        self._Vnn = Vnn

    @property
    def n_basis(self):
        """ number of basis functions """
        return self._n_basis

    @property
    def S(self):
        """ overlap matrix """
        return self._S

    @property
    def Te(self):
        """ kinetic energy of electron """
        return self._Te

    @property
    def Ven(self):
        """ electron-nuclear attraction """
        return self._Ven

    @property
    def Vee(self):
        """ electron-electron repulsion """
        return self._Vee

    @property
    def Vnn(self):
        """The Vnn property."""
        return self._Vnn


class STO2G:

    def __init__(self, atom, xyz):
        self._xyz = xyz
        
        if atom == "Cu":
            self._nprimitive = 2
            self._alpha = [0.6889795561E+03, 0.1226380137E+03]
            self._coeff = [0.4301284983E+00, 0.6789135305E+00]
            self._norm = [(2.0 * a / np.pi)**0.75 for a in self.alpha]
            self._l123 = [0, 0, 0]

    @property
    def xyz(self):
        """ nuclear coordinate """
        return self._xyz
    @property
    def nprimitive(self):
        """ number of primitive gaussians """
        return self._nprimitive
    @property
    def alpha(self):
        """ gaussian exponents """
        return self._alpha
    @property
    def coeff(self):
        """ contraction coefficients """
        return self._coeff
    @property
    def norm(self):
        """ normalization constant """
        return self._norm
    @property
    def l123(self):
        """ angular momentums """
        return self._l123


class STO3G:

    def __init__(self, atom, xyz):
        self._xyz = xyz
        
        if atom == "H":
            self._nprimitive = 3
            self._alpha = [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]
            self._coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
            self._norm = [(2.0 * a / np.pi)**0.75 for a in self.alpha]
            self._l123 = [0, 0, 0]

        elif atom == "He":
            self._nprimitive = 3
            self._alpha = [0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00]
            self._coeff = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
            self._norm = [(2.0 * a / np.pi)**0.75 for a in self.alpha]
            self._l123 = [0, 0, 0]

    @property
    def xyz(self):
        """ nuclear coordinate """
        return self._xyz
    @property
    def nprimitive(self):
        """ number of primitive gaussians """
        return self._nprimitive
    @property
    def alpha(self):
        """ gaussian exponents """
        return self._alpha
    @property
    def coeff(self):
        """ contraction coefficients """
        return self._coeff
    @property
    def norm(self):
        """ normalization constant """
        return self._norm
    @property
    def l123(self):
        """ angular momentums """
        return self._l123

class Six31G:
    def __init__(self, atom, xyz, idx):
        self._xyz = xyz

        if atom == "H":
            if idx == 0:
                self._nprimitive = 3
                self._alpha = [0.1873113696E+02, 0.2825394365E+01, 0.6401216923E+00]
                self._coeff = [0.3349460434E-01, 0.2347269535E+00, 0.8137573261E+00]
            else:
                self._nprimitive = 1
                self._alpha = [0.1612777588E+00]
                self._coeff = [1.0000000]
            
            self._l123 = [0, 0, 0]
            self._norm = [(2.0 * a / np.pi)**0.75 for a in self.alpha]

        elif atom == "He":
            if idx == 0:
                self._nprimitive = 3
                self._alpha = [0.3842163400E+02, 0.5778030000E+01, 0.1241774000E+01]
                self._coeff = [0.4013973935E-01, 0.2612460970E+00, 0.7931846246E+00]
            else:
                self._nprimitive = 1
                self._alpha = [0.2979640000E+00]
                self._coeff = [1.0000000]
            
            self._l123 = [0, 0, 0]
            self._norm = [(2.0 * a / np.pi)**0.75 for a in self.alpha]


    @property
    def xyz(self):
        """ nuclear coordinate """
        return self._xyz
    @property
    def nprimitive(self):
        """ number of primitive gaussians """
        return self._nprimitive
    @property
    def alpha(self):
        """ gaussian exponents """
        return self._alpha
    @property
    def coeff(self):
        """ contraction coefficients """
        return self._coeff
    @property
    def norm(self):
        """ normalization constant """
        return self._norm
    @property
    def l123(self):
        """ angular momentums """
        return self._l123

if __name__ == "__main__":
    print("yongbin")

