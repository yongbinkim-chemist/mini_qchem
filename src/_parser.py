import numpy as np

proton = {"1":"H", "2":"He", "3":"Li", "4":"Be", "5":"B", "6":"C", "7":"N", "8":"O",
          "9":"F", "10":"Ne", "11":"Na", "12":"Mg", "13":"Al", "14":"Si", "15":"P",
          "16":"S", "17":"Cl", "18":"Ar", "19":"Ca", "20":"K"}

def read_fchk_file(ifile):
    line = ifile.read()

    start = line.find("SP")
    basis = line[start:].split()[2]

    start = line.find("Charge")
    charge = int(line[start:].split()[2])

    start = line.find("Multiplicity")
    multiplicity = int(line[start:].split()[2])
    
    start = line.find("Number of electrons")
    n_electrons = int(line[start:].split()[4])

    start = line.find("Atomic numbers")
    mid = line.find("Current cartesian coordinates")
    end = line.find("Nuclear charges")

    atoms = line[start:mid].split()[5:]
    xyz = list(map(float,line[mid:end].split()[6:]))
    geometry = [(proton.get(atoms[i]), np.array(xyz[3*i:3*i+3])) for i in range(len(atoms))]
    
    start = line.find("Number of basis functions")
    n_basis = int(line[start:].split()[5])

    return (basis, n_basis, charge, multiplicity, n_electrons, geometry)
    
    # start = line.find("Core Hamiltonian Matrix")
    # end = line.find("Orbital Coefficients CCMAN2 Alpha")
    # hcore = list(map(float,line[start:end].split()[6:]))
    # Hcore = np.zeros((n_basis,n_basis))
    # k = 0
    # for i in range(n_basis):
    #     for j in range(i+1):
    #         Hcore[i,j] = hcore[j+k]
    #         Hcore[j,i] = hcore[j+k]
    #     k += i+1
    #
    # start = line.find("Alpha MO coefficients")
    # end = line.find("Alpha Orbital Energies")
    # orbs = np.array(list(map(float,line[start:end].split()[6:]))).reshape(n_basis,n_basis)
    #
    # return (basis, n_basis, charge, multiplicity, n_electrons, geometry, Hcore, orbs)

def read_out_file(ifile):
    line = ifile.read()
    start = line.find("Nuclear Repulsion Energy")
    return float(line[start:].split()[4])

def read_mo_ene_file(ifile):
    line = ifile.read()

    start = line.find("O")
    end = line.find("V")

    n_occ = int(line[start:end].split()[2])
    n_vir = int(line[end:].split()[2])

    e_occ = list(map(float,line[start:end].split()[3:3+n_occ//2]))
    e_vir = list(map(float,line[end:].split()[3:3+n_vir//2]))

    return e_occ, e_vir

def read_fock_operator(ifile,mo_map):
    line = ifile.read()

    start = line.find("F_oo")
    mid = line.find("F_ov")
    end = line.find("F_vv")

    o = int(line[start:mid].split()[3])
    v = int(line[mid:end].split()[4])
    foo = list(map(float, line[start:mid].split()[5:]))
    fov = list(map(float, line[mid:end].split()[5:]))
    fvv = list(map(float, line[end:].split()[5:]))

    fock = np.zeros((o+v,o+v))
    fock = get_one_body(mo_map,o,o,0,0,foo,fock)
    fock = get_one_body(mo_map,o,v,0,o,fov,fock)
    fock = get_one_body(mo_map,v,v,o,o,fvv,fock)
    
    return fock

def read_two_body_operator(ifile,mo_map):
    line = ifile.read()
 
    oooo = line.find("OOOO")
    ooov = line.find("OOOV")
    oovv = line.find("OOVV")
    ovov = line.find("OVOV")
    ovvv = line.find("OVVV")
    vvvv = line.find("VVVV")
    
    o = int(line[oooo:ooov].split()[2])
    v = int(line[ooov:oovv].split()[5])

    two_body = np.zeros((o+v,o+v,o+v,o+v))
    two_body = get_two_body(mo_map,o,o,o,o,0,0,0,0,line[oooo:ooov].split()[6:],two_body)
    two_body = get_two_body(mo_map,o,o,o,v,0,0,0,o,line[ooov:oovv].split()[6:],two_body,permute='ooov')
    two_body = get_two_body(mo_map,o,o,v,v,0,0,o,o,line[oovv:ovov].split()[6:],two_body,permute='oovv')
    two_body = get_two_body(mo_map,o,v,o,v,0,o,0,o,line[ovov:ovvv].split()[6:],two_body,permute='ovov')
    two_body = get_two_body(mo_map,o,v,v,v,0,o,o,o,line[ovvv:vvvv].split()[6:],two_body,permute='ovvv')
    two_body = get_two_body(mo_map,v,v,v,v,o,o,o,o,line[vvvv:].split()[6:],two_body)

    return two_body

def get_one_body(mo_map,pmax,qmax,pmin,qmin,raw,fock):
    """
    - 1D to <p|f|q>
    - aaa...abbb...b aaa...abbb...b -> ababab...ab
    """
    for p in range(pmax):
        for q in range(qmax):
            idx = p*qmax + q
            if abs(raw[idx]) >= 10**-16:
                fock[mo_map.get(p+pmin),mo_map.get(q+qmin)] += raw[idx]
    return fock
 
def get_two_body(mo_map,pmax,qmax,rmax,smax,pmin,qmin,rmin,smin,raw,two_body,permute=None):
    """
    - 1D to anti-symmetrized <pq||rs>
    - aaa...abbb...b aaa...abbb...bbb  -> ababab...ab
    - permutations:
        OOOO and VVVV no need permutations
        OOOV = <pq||rs> -> -<pq||sr>, OOVO -> +<rs||pq>, OVOO -> -<sr||pq>, VOOO
        OOVV = <pq||rs> -> +<rs||pq>, VVOO
        OVOV = <pq||rs> -> -<pq||sr>, OVVO -> +<qp||sr>, VOVO -> -<qp||rs>, VOOV
        OVVV = <pq||rs> -> -<qp||rs>, VOVV -> +<rs||pq>, VVOV -> -<rs||qp>, VVVO
    """
    for p in range(pmax):
        for q in range(qmax):
            for r in range(rmax):
                for s in range(smax):
                    # correct orbital order
                    i,j,k,l = mo_map.get(p+pmin),mo_map.get(q+qmin),mo_map.get(r+rmin),mo_map.get(s+smin)
                    idx = (p*qmax*rmax*smax) + (q*rmax*smax) + (r*smax) + s
                    if abs(float(raw[idx])) >= 10**-16:
                        two_body[i,j,k,l] += float(raw[idx])
                        if permute == 'ooov':
                            two_body[i,j,l,k] += -1.0*float(raw[idx])
                            two_body[k,l,i,j] += +1.0*float(raw[idx])
                            two_body[l,k,i,j] += -1.0*float(raw[idx])
                        elif permute == 'oovv':
                            two_body[k,l,i,j] += +1.0*float(raw[idx])
                        elif permute == 'ovov':
                            two_body[i,j,l,k] += -1.0*float(raw[idx])
                            two_body[j,i,l,k] += +1.0*float(raw[idx])
                            two_body[j,i,k,l] += -1.0*float(raw[idx])
                        elif permute == 'ovvv':
                            two_body[j,i,k,l] += -1.0*float(raw[idx])
                            two_body[k,l,i,j] += +1.0*float(raw[idx])
                            two_body[k,l,j,i] += -1.0*float(raw[idx])
    return two_body
