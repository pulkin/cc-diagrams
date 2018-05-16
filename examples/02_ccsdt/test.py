from raw import equations_s, equations_sd, equations_sdt
from pyscf_helpers import kernel, res2amps, eris_hamiltonian

from pyscf import gto, scf
from pyscf.cc.uccsd_slow import UCCSD, update_amps

import numpy
from numpy import testing
import unittest


class H2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "H 0 0 0; H 0.74 0 0"
        cls.mol.basis = 'ccpvdz'
        cls.mol.build()

        cls.mf = scf.UHF(cls.mol)
        cls.mf.kernel()

        cls.ccsd = UCCSD(cls.mf)
        cls.ccsd.kernel()

        cls.eris = cls.ccsd.ao2mo()

    def test_equations(self):
        """Tests coupled-cluster singles and doubles equations."""
        _, t1, t2 = self.ccsd.init_amps(self.eris)

        ref_t1, ref_t2 = update_amps(self.ccsd, t1, t2, self.eris)

        hamiltonian = eris_hamiltonian(self.eris)
        r1, r2, e2 = equations_sd(t1=t1, t2=t2, **hamiltonian)
        dt1, dt2 = res2amps((r1, r2), numpy.diag(hamiltonian["oo"]), numpy.diag(hamiltonian["vv"]))
        t1 += dt1
        t2 += dt2

        testing.assert_allclose(t1, ref_t1, atol=1e-8)
        testing.assert_allclose(t2, ref_t2, atol=1e-8)

    def test_iter_s(self):
        """CCS iterations."""
        ampl = dict(t1=0)
        ampl, e1 = kernel(ampl, self.eris, equations_s)

        testing.assert_allclose(e1, 0, atol=1e-8)
        # TODO: atol=1e-8 does not work
        testing.assert_allclose(ampl["t1"], 0, atol=1e-6)

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(t1=0, t2=0)
        ampl, e2 = kernel(ampl, self.eris, equations_sd)

        testing.assert_allclose(e2, self.ccsd.e_corr)
        testing.assert_allclose(ampl["t1"], self.ccsd.t1, atol=1e-8)
        testing.assert_allclose(ampl["t2"], self.ccsd.t2, atol=1e-8)

    def test_iter_sdt(self):
        """CCSDT iterations (there are no triple excitations for a 2-electron system)."""
        ampl = dict(t1=0, t2=0, t3=0)
        ampl, e3 = kernel(ampl, self.eris, equations_sdt)

        testing.assert_allclose(e3, self.ccsd.e_corr, atol=1e-8)
        testing.assert_allclose(ampl["t1"], self.ccsd.t1, atol=1e-8)
        testing.assert_allclose(ampl["t2"], self.ccsd.t2, atol=1e-8)
        testing.assert_allclose(ampl["t3"], 0, atol=1e-8)


class OTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Oxygen atom test vs ORCA-MRCC data.

        ORCA version 3.0.3

        Input example:

        ! cc-pvdz UHF TightSCF

        %mrcc
          method "CCSDT"
          ETol 10
        end

        %pal nprocs 4
        end

        * xyzfile 0 1 initial.xyz

        ORCA reference energies:

        HF    -74.6652538779
        CCS   -74.841686696943
        CCSD  -74.819248718982
        CCSDT -74.829163218204
        """
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "O 0 0 0"
        cls.mol.basis = 'cc-pvdz'
        cls.mol.build()

        cls.mf = scf.UHF(cls.mol)
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()
        testing.assert_allclose(cls.mf.e_tot, -74.6652538779, atol=1e-4)

        cls.ccsd = UCCSD(cls.mf, frozen=1)
        cls.ccsd.kernel()

        cls.eris = cls.ccsd.ao2mo()

    def test_iter_s(self):
        """CCS iterations."""
        ampl = dict(t1=0)
        ampl, e1 = kernel(ampl, self.eris, equations_s, tolerance=1e-6)
        # TODO: MRCC energy is way off expected
        testing.assert_allclose(self.mf.e_tot + e1, -74.841686696943, atol=1e-4)

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(t1=0, t2=0)
        ampl, e2 = kernel(ampl, self.eris, equations_sd, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e2, -74.819248718982, atol=1e-4)

    def test_iter_sdt(self):
        """CCSDT iterations."""
        ampl = dict(t1=0, t2=0, t3=0)
        ampl, e3 = kernel(ampl, self.eris, equations_sdt, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e3, -74.829163218204, atol=1e-4)


class H2OTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        H20 molecule test vs ORCA-MRCC data.

        ORCA reference energies:

        HF    -75.97354725
        CCS   --
        CCSD  -76.185805898396
        CCSDT -76.189327633478
        """
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "O 0 0 0; H  0.758602  0.000000  0.504284; H  0.758602  0.000000  -0.504284"
        cls.mol.unit = "angstrom"

        cls.mol.basis = 'cc-pvdz'
        cls.mol.build()

        cls.mf = scf.UHF(cls.mol)
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()
        testing.assert_allclose(cls.mf.e_tot, -75.97354725, atol=1e-4)

        cls.ccsd = UCCSD(cls.mf, frozen=1)
        cls.ccsd.kernel()

        cls.eris = cls.ccsd.ao2mo()

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(t1=0, t2=0)
        ampl, e2 = kernel(ampl, self.eris, equations_sd, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e2, -76.185805898396, atol=1e-4)

    def test_iter_sdt(self):
        """CCSDT iterations."""
        ampl = dict(t1=0, t2=0, t3=0)
        ampl, e3 = kernel(ampl, self.eris, equations_sdt, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e3, -76.189327633478, atol=1e-4)
