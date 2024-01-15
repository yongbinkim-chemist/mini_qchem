.. <div align="left">
..   <img src="https://github.com/yongbinkim-chemist/logo/logo.png" height="80px"/>
.. </div>

100% Python-based cute ab-initio quantum chemistry program
=================

The mini_qchem library is a 100% Python-based ab-initio quantum chemistry calculation program.
The purpose of this project is to enhance my personal understanding of electronic structure theory in practice.
Currently, I am working on the Davidson algorithm for equation-of-motion coupled-cluster, and I plan to release the source code and jupyter-notebook demo right after the Davidson implementation.

As a theoretical and computational quantum chemist with a passion for programming, I enjoy receiving feedback from the community.
If there are any areas for improvement, I welcome suggestions at any time.

Feel free to share your thoughts or ask any questions about the project `chem.yongbin@gmail.come <chem.yongbin@gmail.com>`__.
I look forward to engaging with the outside the world and contributing to interesting projects.

Log (01/14/2024)
------------
- Source codes will be released after Davidson algorithm implementation for EOM-CCSD

Available features 
------------
- Hartree-Fock 
- MP2 and MP3
- CCSD
- EOM-CCSD -> Done | Davidson -> in progress...

Requirements
------------
- Python 3+
- Numpy
- Scipy

Installation
------------
To use mini-qchem:

.. code-block:: bash

  git clone https://github.com/yongbinkim-chemist/mini_qchem.git 
  cd mini_qchem
  python -m pip install -e .

Please take a look at the `ipython notebook demo <https://github.com/demo/mini_qchem.ipynb>`__.

Authors
-------

`Yongbin Kim <https://github.com/yongbinkim-chemist>`__ (University of Southern California),
