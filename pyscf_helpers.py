from pyscf.lib.diis import DIIS
import numpy

import inspect
from numbers import Number


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
        if isinstance(i, Number) and i == 0:
            result.append(0)
        else:
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
    """Amplitudes into a single array."""
    result = []
    for k in sorted(amplitudes.keys()):
        result.append(numpy.reshape(amplitudes[k], -1))
    return numpy.concatenate(result)


def v2a(vec, amplitudes):
    """Array into a dict of amplitudes."""
    result = {}
    offset = 0
    for k in sorted(amplitudes.keys()):
        s = amplitudes[k].size
        result[k] = numpy.reshape(vec[offset:offset+s], amplitudes[k].shape)
        offset += s
    return result


def eris_hamiltonian(eris):
    """
    Retrieves Hamiltonian matrix elements from pyscf ERIS.
    Args:
        eris (pyscf.cc.ccsd.ERIS): pyscf ERIS;

    Returns:
        A dict with Hamiltonian matrix elements.
    """
    nocc = eris.oooo.shape[0]
    return dict(
        ov=eris.fock[:nocc, nocc:],
        vo=eris.fock[nocc:, :nocc],
        oo=eris.fock[:nocc, :nocc],
        vv=eris.fock[nocc:, nocc:],
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


def kernel(amplitudes, hamiltonian, equations, tolerance=1e-9, debug=False, diis=True):
    """
    Coupled-cluster iterations.
    Args:
        amplitudes (dict): starting amplitudes;
        hamiltonian (dict): hamiltonian matrix elements or pyscf ERIS;
        equations (callable): coupled-cluster equations;
        tolerance (float): convergence criterion;
        debug (bool): prints iterations if True;
        diis (bool, DIIS): converger for iterations;

    Returns:
        Resulting coupled-cluster amplitudes and energy.
    """
    if not isinstance(hamiltonian, dict):
        hamiltonian = eris_hamiltonian(hamiltonian)

    tol = None
    e_occ = numpy.diag(hamiltonian["oo"])
    e_vir = numpy.diag(hamiltonian["vv"])
    if diis is True:
        diis = DIIS()

    input_args = inspect.getargspec(equations).args
    hamiltonian = {k: v for k, v in hamiltonian.items() if k in input_args}

    while tol is None or tol > tolerance:
        hamiltonian.update(amplitudes)
        output = equations(**hamiltonian)
        e_corr = output[-1]
        dt = res2amps(output[:-1], e_occ, e_vir)
        tol = max(numpy.linalg.norm(i) for i in dt)
        for k, delta in zip(sorted(amplitudes), dt):
            amplitudes[k] = amplitudes[k] + delta

        if diis and not any(isinstance(i, Number) for i in amplitudes.values()):
            v = a2v(amplitudes)
            amplitudes = v2a(diis.update(v), amplitudes)

        if debug:
            print("E_corr = {:.15f}, dt={:.3e}".format(e_corr, tol))

    return amplitudes, e_corr
