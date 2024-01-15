import numpy as np
from src import MolecularData, SCF, MBPT, CCSD, EOMCCSD

def run(molecule, spin='singlet', run_scf=True, run_mp2=False, run_ccsd=False, run_eom_ccsd=False):

    if run_scf:
        hf = SCF(molecule)
        hf.run()

    if run_mp2:
        mbpt = MBPT(molecule)
        mbpt.run(order=2)

    if run_ccsd:
        ccsd = CCSD(molecule)
        ccsd.run_ccsd()

    if run_eom_ccsd:
        eom_ccsd = EOMCCSD(molecule)
        eom_ccsd.run_eom(spin=spin)


if __name__ == "__main__":

    # path = "/Users/yongbinkim/Desktop/venv/qchem/mini_qchem/src/examples/H2/sto-3g/0.8/"
    # path = "/Users/yongbinkim/Desktop/venv/qchem/mini_qchem/src/examples/H2/6-31g/0.8/"
    # path = "/Users/yongbinkim/Desktop/venv/qchem/mini_qchem/src/examples/H4/2.00/"
    # path = "/Users/yongbinkim/Desktop/venv/qchem/mini_qchem/src/examples/LiH/1.4/"
    # path = "/Users/yongbinkim/Desktop/venv/qchem/mini_qchem/src/examples/H2O/1.4/"
    # molecule1 = MolecularData(run=False,path=path)
    # run(molecule1,spin='triplet',run_scf=False,run_ccsd=True)
    # run(molecule1,spin='triplet',run_scf=False,run_eom_ccsd=True)
    
    basis = 'sto-2g'

    geometry = [('Cu', np.array([-1.3544366929, -1.4339147394, 0.2068509192])),
                ('Cu', np.array([-0.5616993292,  1.8925214655, 0.1830332929])),
                ('Cu', np.array([ 1.9197039511, -0.4585599885, 0.1909596008]))]

    # geometry = [('H', np.array([0., 0., 0.])), ('H', np.array([0., 0., 1.2]))]
    # geometry = [('Li', np.array([0., 0., 0.])), ('H', np.array([0., 0., 1.6]))]
    multiplicity = 1
    charge = 0
    molecule2 = MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
    run(molecule2)
    # run(molecule2,run_ccsd=True)
