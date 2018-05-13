from raw import equations_s, equations_sd, equations_sdt

from pyscf import gto, scf
from pyscf.cc.uccsd_slow import UCCSD, update_amps
from pyscf.lib.diis import DIIS

import numpy
from numpy import testing
import unittest


def res2amps(residuals, e_occ, e_vir):
    """
    Converts residuals into amplitudes update.
    Args:
        residuals (iterable): a list of residuals (pyscf index order);
        e_occ (array): occupied energies;
        e_vir (array): virtual energies;

    Returns:
        A list of updates to amplitudes.
    """
    result = []
    for i in residuals:
        order = len(i.shape) // 2
        diagonal = numpy.zeros_like(i)

        for j in range(order):
            ix = [numpy.newaxis] * (2*order)
            ix[j] = slice(None)
            diagonal += e_occ[ix]

            ix[j] = numpy.newaxis
            ix[j+order] = slice(None)
            diagonal -= e_vir[ix]

        result.append(i / diagonal)
    return result


def a2v(amplitudes):
    result = []
    for k in sorted(amplitudes.keys()):
        result.append(numpy.reshape(amplitudes[k], -1))
    return numpy.concatenate(result)


def v2a(vec, amplitudes):
    result = {}
    offset = 0
    for k in sorted(amplitudes.keys()):
        s = amplitudes[k].size
        result[k] = numpy.reshape(vec[offset:offset+s], amplitudes[k].shape)
        offset += s
    return result


def kernel(amplitudes, hamiltonian, equations, tolerance=1e-9, debug=False):
    """
    Coupled-cluster iterations.
    Args:
        amplitudes (dict): starting amplitudes;
        hamiltonian (dict): hamiltonian matrix elements;
        equations (callable): coupled-cluster equations;
        tolerance (float): convergence criterion;
        debug (bool): prints iterations if True;

    Returns:
        Resulting coupled-cluster amplitudes and energy.
    """
    tol = None
    e_occ = numpy.diag(hamiltonian["oo"])
    e_vir = numpy.diag(hamiltonian["vv"])
    adiis = DIIS()

    while tol is None or tol > tolerance:
        hamiltonian.update(amplitudes)
        output = equations(**hamiltonian)
        e_corr = output[-1]
        dt = res2amps(output[:-1], e_occ, e_vir)
        tol = max(numpy.linalg.norm(i) for i in dt)
        for k, delta in zip(sorted(amplitudes), dt):
            amplitudes[k] = amplitudes[k] + delta
        v = a2v(amplitudes)
        amplitudes = v2a(adiis.update(v), amplitudes)

        if debug:
            print("E_corr = {:.15f}, dt={:.3e}".format(e_corr, tol))

    return amplitudes, e_corr


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

        cls.nocc, cls.nvir = cls.ccsd.t1.shape
        cls.eris = cls.ccsd.ao2mo()

        cls.hamiltonian = dict(
            ov=cls.eris.fock[:cls.nocc, cls.nocc:],
            vo=cls.eris.fock[cls.nocc:, :cls.nocc],
            oo=cls.eris.fock[:cls.nocc, :cls.nocc],
            vv=cls.eris.fock[cls.nocc:, cls.nocc:],
            oooo=cls.eris.oooo,
            oovo=-numpy.transpose(cls.eris.ooov, (0, 1, 3, 2)),
            oovv=cls.eris.oovv,
            ovoo=cls.eris.ovoo,
            ovvo=-numpy.transpose(cls.eris.ovov, (0, 1, 3, 2)),
            ovvv=cls.eris.ovvv,
            vvoo=numpy.transpose(cls.eris.oovv, (2, 3, 0, 1)),
            vvvo=-numpy.transpose(cls.eris.ovvv, (2, 3, 1, 0)),
            vvvv=cls.eris.vvvv,
        )

        cls.e_occ = numpy.diag(cls.hamiltonian["oo"])
        cls.e_vir = numpy.diag(cls.hamiltonian["vv"])

    def test_equations(self):
        """Tests coupled-cluster singles and doubles equations."""
        _, t1, t2 = self.ccsd.init_amps(self.eris)

        ref_t1, ref_t2 = update_amps(self.ccsd, t1, t2, self.eris)

        r1, r2, e2 = equations_sd(t1=t1, t2=t2, **self.hamiltonian)
        dt1, dt2 = res2amps((r1, r2), numpy.diag(self.hamiltonian["oo"]), numpy.diag(self.hamiltonian["vv"]))
        t1 += dt1
        t2 += dt2

        testing.assert_allclose(t1, ref_t1, atol=1e-8)
        testing.assert_allclose(t2, ref_t2, atol=1e-8)

    def test_iter_s(self):
        """CCS iterations."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
        )
        ham = dict((k, v) for k, v in self.hamiltonian.items() if k in (
                "oo", "ov", "vo", "vv", "oovo", "oovv", "ovvo", "ovvv"))
        ampl, e1 = kernel(ampl, ham, equations_s)

        testing.assert_allclose(e1, 0, atol=1e-8)
        # TODO: atol=1e-8 does not work
        testing.assert_allclose(ampl["t1"], 0, atol=1e-6)

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
            t2=numpy.zeros((self.nocc, self.nocc, self.nvir, self.nvir)),
        )
        ampl, e2 = kernel(ampl, self.hamiltonian, equations_sd)

        testing.assert_allclose(e2, self.ccsd.e_corr)
        testing.assert_allclose(ampl["t1"], self.ccsd.t1, atol=1e-8)
        testing.assert_allclose(ampl["t2"], self.ccsd.t2, atol=1e-8)

    def test_iter_sdt(self):
        """CCSDT iterations (there are no triple excitations for a 2-electron system)."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
            t2=numpy.zeros((self.nocc, self.nocc, self.nvir, self.nvir)),
            t3=numpy.zeros((self.nocc, self.nocc, self.nocc, self.nvir, self.nvir, self.nvir)),
        )
        ampl, e3 = kernel(ampl, self.hamiltonian, equations_sdt)

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

        cls.nocc, cls.nvir = cls.ccsd.t1.shape
        eris = cls.ccsd.ao2mo()

        cls.hamiltonian = dict(
            ov=eris.fock[:cls.nocc, cls.nocc:],
            vo=eris.fock[cls.nocc:, :cls.nocc],
            oo=eris.fock[:cls.nocc, :cls.nocc],
            vv=eris.fock[cls.nocc:, cls.nocc:],
            oooo=eris.oooo,
            oovo=-numpy.transpose(eris.ooov, (0, 1, 3, 2)),
            oovv=eris.oovv,
            ovoo=eris.ovoo,
            ovvo=-numpy.transpose(eris.ovov, (0, 1, 3, 2)),
            ovvv=eris.ovvv,
            vvoo=numpy.transpose(eris.oovv, (2, 3, 0, 1)),
            vvvo=-numpy.transpose(eris.ovvv, (2, 3, 1, 0)),
            vvvv=eris.vvvv,
        )

        cls.e_occ = numpy.diag(cls.hamiltonian["oo"])
        cls.e_vir = numpy.diag(cls.hamiltonian["vv"])

    def test_iter_s(self):
        """CCS iterations."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
        )
        ham = dict((k, v) for k, v in self.hamiltonian.items() if k in (
                "oo", "ov", "vo", "vv", "oovo", "oovv", "ovvo", "ovvv"))
        ampl, e1 = kernel(ampl, ham, equations_s, tolerance=1e-6)
        # TODO: MRCC energy is way off expected
        testing.assert_allclose(self.mf.e_tot + e1, -74.841686696943, atol=1e-4)

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
            t2=numpy.zeros((self.nocc, self.nocc, self.nvir, self.nvir)),
        )
        ampl, e2 = kernel(ampl, self.hamiltonian, equations_sd, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e2, -74.819248718982, atol=1e-4)

    def test_iter_sdt(self):
        """CCSDT iterations."""
        ampl = dict(
            t1=self.ccsd.t1,
            t2=self.ccsd.t2,
            t3=numpy.zeros((self.nocc, self.nocc, self.nocc, self.nvir, self.nvir, self.nvir)),
        )
        ampl, e3 = kernel(ampl, self.hamiltonian, equations_sdt, tolerance=1e-6)
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

        cls.nocc, cls.nvir = cls.ccsd.t1.shape
        eris = cls.ccsd.ao2mo()

        cls.hamiltonian = dict(
            ov=eris.fock[:cls.nocc, cls.nocc:],
            vo=eris.fock[cls.nocc:, :cls.nocc],
            oo=eris.fock[:cls.nocc, :cls.nocc],
            vv=eris.fock[cls.nocc:, cls.nocc:],
            oooo=eris.oooo,
            oovo=-numpy.transpose(eris.ooov, (0, 1, 3, 2)),
            oovv=eris.oovv,
            ovoo=eris.ovoo,
            ovvo=-numpy.transpose(eris.ovov, (0, 1, 3, 2)),
            ovvv=eris.ovvv,
            vvoo=numpy.transpose(eris.oovv, (2, 3, 0, 1)),
            vvvo=-numpy.transpose(eris.ovvv, (2, 3, 1, 0)),
            vvvv=eris.vvvv,
        )

        cls.e_occ = numpy.diag(cls.hamiltonian["oo"])
        cls.e_vir = numpy.diag(cls.hamiltonian["vv"])

    def test_iter_sd(self):
        """CCSD iterations."""
        ampl = dict(
            t1=numpy.zeros((self.nocc, self.nvir)),
            t2=numpy.zeros((self.nocc, self.nocc, self.nvir, self.nvir)),
        )
        ampl, e2 = kernel(ampl, self.hamiltonian, equations_sd, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e2, -76.185805898396, atol=1e-4)

    def test_iter_sdt(self):
        """CCSDT iterations."""
        ampl = dict(
            t1=self.ccsd.t1,
            t2=self.ccsd.t2,
            t3=numpy.zeros((self.nocc, self.nocc, self.nocc, self.nvir, self.nvir, self.nvir)),
        )
        ampl, e3 = kernel(ampl, self.hamiltonian, equations_sdt, tolerance=1e-6)
        testing.assert_allclose(self.mf.e_tot + e3, -76.189327633478, atol=1e-4)
