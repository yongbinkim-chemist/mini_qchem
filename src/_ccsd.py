import numpy as np
import time

class CCSD():
    """
    CCSD amplitudes, intermediate approach
    Hbar = e^(-T) Hn e^(-T) = {Hn e^(-T)}c
    It is done by Yongbin Kim based on the Q-CHEM libcc
    Reference for Equations:
        Many-Body Methods in Chemistry and Physics by Shavitt and Bartlett (Chap 10.7 and Chap 13)
        Gauss, et al  : J. Chem. Phys. 103, 3561 (1995) https://doi.org/10.1063/1.470240
        Stanton, et al: J. Chem. Phys.  98, 7029 (1993) https://doi.org/10.1063/1.464746
        Salter, et al : J. Chem. Phys.  90, 1752 (1989) https://doi.org/10.1063/1.456069
        Gauss, et al  : J. Chem. Phys.  95, 2623 (1991) https://doi.org/10.1063/1.460915
        Levchenko et al J. Chem. Phys. 120,  175 (2004) https://doi.org/10.1063/1.1630018
    """
    def __init__(self,molecule):
        o,v = molecule.n_electrons, molecule.n_orbitals - molecule.n_electrons
        self.n_occ = o
        self.n_vir = v
        self.hf_energy = molecule.hf_energy
        self.f_oo = molecule.fock_operator[:o, :o]
        self.f_ov = molecule.fock_operator[:o, o:o+v]
        self.f_vv = molecule.fock_operator[o:o+v, o:o+v]
        self.i_oooo = molecule.two_body_integrals[:o, :o, :o, :o]
        self.i_ooov = molecule.two_body_integrals[:o, :o, :o, o:o+v]
        self.i_oovv = molecule.two_body_integrals[:o, :o, o:o+v, o:o+v]
        self.i_ovov = molecule.two_body_integrals[:o, o:o+v, :o, o:o+v]
        self.i_ovvv = molecule.two_body_integrals[:o, o:o+v, o:o+v, o:o+v]
        self.i_vvvv = molecule.two_body_integrals[o:o+v, o:o+v, o:o+v, o:o+v]
        self.thresh = 1e-12
        self.molecule = molecule
        self.t1 = np.zeros((o,v))

    # def __del__(self):
    #     print("Destroy CCSD instance")

    def run_ccsd(self,e_conv=1e-6,maxiter=100):
        self.ccsd_start = time.time()
        conv = False

        ############
        # RUN CCSD #
        ############
        self.compute_diag()
        self.i_guess()
        self.corr = self.compute_energy()
        self.mp2_energy = self.corr + self.hf_energy
        self.molecule._mp2_energy = self.corr

        print('------------------------------------------------')
        print('            Energy (a.u.)    Ediff')
        print('------------------------------------------------')
        for iter in range(1, maxiter+1):
            self.build_ccsd_intermediate()
            t1new = self.t1_amps()
            t2new = self.t2_amps()
            self.t1 = t1new
            self.t2 = t2new
            corr_new = self.compute_energy()

            if abs(corr_new-self.corr) < e_conv:
                print("{0:5d} {1:17.8f}* {2:13.5e}".format(iter, corr_new+self.hf_energy, abs(corr_new-self.corr)))
                self.corr = corr_new
                self.molecule._ccsd_energy = self.corr
                conv = True
                break
            print("{0:5d} {1:17.8f} {2:14.5e}".format(iter, corr_new+self.hf_energy, abs(corr_new-self.corr)))
            self.corr = corr_new

        if conv:
            print('------------------------------------------------')
            print("SCF energy                 = {:.8f}".format(self.hf_energy))
            print("MP2 energy                 = {:.8f}".format(self.mp2_energy))
            print("CCSD correlation energy    = {:.8f}".format(self.corr))
            print("CCSD total energy          = {:.8f}".format(self.corr+self.hf_energy))
            print("\nCCSD calculation: {:.2f} s".format(time.time()-self.ccsd_start))
            print('------------------------------------------------')
        else:
           raise Exception("CCSD Failed -- Iterations exceeded")

    def compute_diag(self):
        d_ov   = np.zeros((self.n_occ,self.n_vir))
        d_oovv = np.zeros((self.n_occ,self.n_occ,self.n_vir,self.n_vir))
        for i in range(self.n_occ):
            for a in range(self.n_vir):
                d_ov[i,a] = self.f_oo[i,i] - self.f_vv[a,a]
                for j in range(self.n_occ):
                    for b in range(self.n_vir):
                        d_oovv[i,j,a,b] = self.f_oo[i,i] + self.f_oo[j,j] - self.f_vv[a,a] - self.f_vv[b,b]
        self.d_ov = d_ov
        self.d_oovv = d_oovv

    def i_guess(self):
        # initial t2 recovers MP2
        self.t2 = np.zeros((self.n_occ,self.n_occ,self.n_vir,self.n_vir))
        non_zero = np.argwhere(abs(self.i_oovv) > self.thresh)
        for ijab in non_zero:
            i,j,a,b = ijab
            self.t2[i,j,a,b] += self.i_oovv[i,j,a,b] / self.d_oovv[i,j,a,b]

    def compute_energy(self):
        """
        Ecc = f_{ia}*t1_{ai} + 0.25*v_{ijab}*t2_{abij} + 0.5*v_{ijab}*t1_{ai}*t1_{bj}
        """
        cc_corr = 0.0
        for i in range(self.n_occ):
            for a in range(self.n_vir):
                # HF reference, off_diag(F) == 0
                cc_corr += self.f_ov[i,a] * self.t1[i,a]
                for j in range(self.n_occ):
                    for b in range(self.n_vir):
                        cc_corr += 0.25 * self.i_oovv[i,j,a,b] * self.t2[i,j,a,b]
                        cc_corr += 0.5  * self.i_oovv[i,j,a,b] * self.t1[i,a] * self.t1[j,b]
        return cc_corr

    #####################
    # CCSD intermediate #
    #####################
    def build_ccsd_intermediate(self):
        self.generate_f1_vv()
        self.generate_f2_ov()
        self.generate_f3_oo()
        self.generate_f2_vv()
        self.generate_tt_oovv()
        self.generate_i4_oooo()
        self.generate_i2a_ooov()
        self.generate_i1a_ovov()
        self.generate_f2_oo()
    
    def generate_f1_vv(self):
        f1_vv = np.zeros((self.n_vir,self.n_vir))
        for b in range(self.n_vir):
            for c in range(self.n_vir):
                f1_vv[b,c] += self.f_vv[b,c]
                for k in range(self.n_occ):
                    for d in range(self.n_vir):
                        f1_vv[b,c] += self.t1[k,d] * self.i_ovvv[k,b,d,c]
                        for l in range(self.n_occ):
                            f1_vv[b,c] -= 0.5 * self.t2[k,l,b,d] * self.i_oovv[k,l,c,d]
        self.f1_vv = f1_vv

    def generate_f2_oo(self):
        f2_oo = np.zeros((self.n_occ,self.n_occ))
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                f2_oo[i,j] += self.f_oo[i,j]
                for a in range(self.n_vir):
                    # HF reference, off_diag(F) == 0
                    f2_oo[i,j] += self.t1[i,a] * self.f_ov[j,a]
                    for k in range(self.n_occ):
                        f2_oo[i,j] += self.t1[k,a] * self.i_ooov[j,k,i,a]
                        for b in range(self.n_vir):
                            f2_oo[i,j] += self.t1[i,a] * self.t1[k,b] * self.i_oovv[j,k,a,b]
                            f2_oo[i,j] += 0.5 * self.t2[i,k,a,b] * self.i_oovv[j,k,a,b]
        self.f2_oo = f2_oo

    def generate_f2_ov(self):
        f2_ov = np.zeros((self.n_occ,self.n_vir))
        for i in range(self.n_occ):
            for a in range(self.n_vir):
                # HF reference, off_diag(F) == 0
                f2_ov[i,a] += self.f_ov[i,a]
                for j in range(self.n_occ):
                    for b in range(self.n_vir):
                        f2_ov[i,a] += self.t1[j,b] * self.i_oovv[i,j,a,b]
        self.f2_ov = f2_ov

    def generate_f2_vv(self):
        f2_vv = np.zeros((self.n_vir,self.n_vir))
        for b in range(self.n_vir):
            for c in range(self.n_vir):
                f2_vv[b,c] += self.f1_vv[b,c]
                for k in range(self.n_occ):
                    if self.t1[k,b] < self.thresh:
                        continue
                    f2_vv[b,c] -= self.t1[k,b] * self.f_ov[k,c]
                    for l in range(self.n_occ):
                        for d in range(self.n_vir):
                            f2_vv[b,c] -= self.t1[k,b] * self.t1[l,d] * self.i_oovv[k,l,c,d]
        self.f2_vv = f2_vv

    def generate_f3_oo(self):
        f3_oo = np.zeros((self.n_occ,self.n_occ))
        for k in range(self.n_occ):
            for i in range(self.n_occ):
                f3_oo[k,i] += self.f_oo[k,i]
                for c in range(self.n_vir):
                    f3_oo[k,i] += self.t1[i,c] * self.f2_ov[k,c]
                    for l in range(self.n_occ):
                        f3_oo[k,i] += self.t1[l,c] * self.i_ooov[k,l,i,c]
                for j in range(self.n_occ):
                    for a in range(self.n_vir):
                        for b in range(self.n_vir):
                            f3_oo[k,i] += 0.5 * self.t2[i,j,a,b] * self.i_oovv[k,j,a,b]
        self.f3_oo = f3_oo

    def generate_i1a_ovov(self):
        i1a_ovov = np.zeros((self.n_occ,self.n_vir,self.n_occ,self.n_vir))
        for i in range(self.n_occ):
            for a in range(self.n_vir):
                for j in range(self.n_occ):
                    for b in range(self.n_vir):
                        i1a_ovov[i,a,j,b] += self.i_ovov[i,a,j,b]
                        for c in range(self.n_vir):
                            i1a_ovov[i,a,j,b] -= self.t1[j,c] * self.i_ovvv[i,a,b,c]
                        for k in range(self.n_occ):
                            i1a_ovov[i,a,j,b] -= self.t1[k,a] * self.i_ooov[i,k,j,b]
                            for c in range(self.n_vir):
                                i1a_ovov[i,a,j,b] -= 0.5 * self.i_oovv[i,k,c,b] * (self.t2[j,k,c,a] + 2.0 * self.t1[j,c] * self.t1[k,a])
        self.i1a_ovov = i1a_ovov

    def generate_tt_oovv(self):
        tt_oovv = np.zeros((self.n_occ,self.n_occ,self.n_vir,self.n_vir))
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for a in range(self.n_vir):
                    for b in range(self.n_vir):
                        tt_oovv[i,j,a,b] += self.t2[i,j,a,b]
                        tt_oovv[i,j,a,b] += 0.5 * (self.t1[i,a]*self.t1[j,b] - self.t1[j,a]*self.t1[i,b]\
                                                   - self.t1[i,b]*self.t1[j,a] + self.t1[j,b]*self.t1[i,a])
        self.tt_oovv = tt_oovv

    def generate_i4_oooo(self):
        i4_oooo = np.zeros((self.n_occ,self.n_occ,self.n_occ,self.n_occ))
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for k in range(self.n_occ):
                    for l in range(self.n_occ):
                        i4_oooo[i,j,k,l] += self.i_oooo[i,j,k,l]
                        for a in range(self.n_vir):
                            i4_oooo[i,j,k,l] += (self.t1[j,a] * self.i_ooov[k,l,i,a] - self.t1[i,a] * self.i_ooov[k,l,j,a])
                            for b in range(self.n_vir):
                                i4_oooo[i,j,k,l] += 0.5 * self.tt_oovv[i,j,a,b] * self.i_oovv[k,l,a,b]
        self.i4_oooo = i4_oooo

    def generate_i2a_ooov(self):
        i2a_ooov = np.zeros((self.n_occ,self.n_occ,self.n_occ,self.n_vir))
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for k in range(self.n_occ):
                    for b in range(self.n_vir):
                        i2a_ooov[i,j,k,b] += self.i_ooov[i,j,k,b]
                        for l in range(self.n_occ):
                            i2a_ooov[i,j,k,b] -= 0.5 * self.t1[l,b] * self.i4_oooo[i,j,k,l]
                        for c in range(self.n_vir):
                            i2a_ooov[i,j,k,b] += (self.t1[j,c] * self.i_ovov[k,b,i,c] - self.t1[i,c] * self.i_ovov[k,b,j,c])
                            for d in range(self.n_vir):
                                i2a_ooov[i,j,k,b] += 0.5 * self.tt_oovv[i,j,c,d] * self.i_ovvv[k,b,c,d]
        self.i2a_ooov = i2a_ooov

    def t1_amps(self):
        t1 = np.zeros((self.n_occ,self.n_vir))
        for a in range(self.n_vir):
            for i in range(self.n_occ):
                t1[i,a] += self.f_ov[i,a]
                for d in range(self.n_vir):
                    t1[i,a] += (self.t1[i,d] * self.f1_vv[a,d] - (a == d) * self.t1[i,d] * self.f_vv[a,d]) 
                for l in range(self.n_occ):
                    t1[i,a] -= (self.t1[l,a] * self.f3_oo[l,i] - (l == i) * self.t1[l,a] * self.f_oo[l,i])
                for k in range(self.n_occ):
                    for c in range(self.n_vir):
                        t1[i,a] -= self.t1[k,c] * self.i_ovov[i,c,k,a]
                        t1[i,a] += self.t2[i,k,a,c] * self.f2_ov[k,c]
                        for d in range(self.n_vir):
                            t1[i,a] += 0.5 * self.t2[k,i,c,d] * self.i_ovvv[k,a,c,d]
                        for l in range(self.n_occ):
                            t1[i,a] -= 0.5 * self.t2[k,l,a,c] * self.i_ooov[k,l,i,c]
                t1[i,a] /= self.d_ov[i,a]
        return t1

    def t2_amps(self):
        t2 = np.zeros((self.n_occ,self.n_occ,self.n_vir,self.n_vir))
        for a in range(self.n_vir):
            for b in range(self.n_vir):
                for i in range(self.n_occ):
                    for j in range(self.n_occ):
                        t2[i,j,a,b] += self.i_oovv[i,j,a,b]
                        for e in range(self.n_vir):
                            t2[i,j,a,b] += self.t2[i,j,a,e] * self.f2_vv[b,e] # I
                            t2[i,j,a,b] -= self.t2[i,j,b,e] * self.f2_vv[a,e] # P(ab)
                            t2[i,j,a,b] -= self.t2[i,j,a,e] * (b == e) * self.f_vv[b,e] # avoid double counting
                            t2[i,j,a,b] += self.t2[i,j,b,e] * (a == e) * self.f_vv[a,e] # avoid double counting
                            t2[i,j,a,b] += self.t1[i,e] * self.i_ovvv[j,e,b,a] # I
                            t2[i,j,a,b] -= self.t1[j,e] * self.i_ovvv[i,e,b,a] # P(ij)
                            for m in range(self.n_occ):
                                t2[i,j,a,b] += self.t2[j,m,a,e] * self.i1a_ovov[m,b,i,e] # I
                                t2[i,j,a,b] -= self.t2[i,m,a,e] * self.i1a_ovov[m,b,j,e] # P(ij)
                                t2[i,j,a,b] -= self.t2[j,m,b,e] * self.i1a_ovov[m,a,i,e] # P(ab)
                                t2[i,j,a,b] += self.t2[i,m,b,e] * self.i1a_ovov[m,a,j,e] # P(ij|ab)
                            for f in range(self.n_vir):
                                t2[i,j,a,b] += 0.5 * self.i_vvvv[a,b,e,f] * self.tt_oovv[i,j,e,f]
                        for m in range(self.n_occ):
                            t2[i,j,a,b] -= self.t2[i,m,a,b] * self.f2_oo[m,j] # I
                            t2[i,j,a,b] += self.t2[j,m,a,b] * self.f2_oo[m,i] # P(ij)
                            t2[i,j,a,b] += self.t2[i,m,a,b] * (m == j) * self.f_oo[m,j] # avoid double counting
                            t2[i,j,a,b] -= self.t2[j,m,a,b] * (m == i) * self.f_oo[m,i] # avoid double counting
                            t2[i,j,a,b] -= self.t1[m,a] * self.i2a_ooov[i,j,m,b] # I
                            t2[i,j,a,b] += self.t1[m,b] * self.i2a_ooov[i,j,m,a] # P(ab)
                            for n in range(self.n_occ):
                                t2[i,j,a,b] += 0.5 * self.t2[m,n,a,b] * self.i4_oooo[m,n,i,j]
                        t2[i,j,a,b] /= self.d_oovv[i,j,a,b]
        return t2
