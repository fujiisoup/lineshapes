import numpy as np
import pyspectra
import py3nj


uB = 5.7883818060e-5  # Bohr magneton in [eV/T]
gL = 1
gS = 2.002319304386
a0 = 5.29e-11  # bohr radius in [m]


def LSzeeman(
    energy_upper, energy_lower,
    J_upper, J_lower,
    L_upper, L_lower,
    S, 
    B,
    line_strengths=None,
    return_xr=False,
):
    r'''
    Calculates Zeeman pattern

    Parameters
    ----------
    energy_upper: np.array with length [n], in eV
        energy of the upper states
    energy_lower: np.array with length [k], in eV
        energy of the upper states
    J_upper: np.array with length [n], integer
        Total angular momentum of the upper states
    J_lower: np.array with length [k], integer
        Total angular momentum of the lower states
    L_upper: np.array with length [n], integer
        Orbital angular momentum of the upper states
    L_lower: np.array with length [k], integer
        Orbital angular momentum of the lower states
    S: Scalar
        Spin angular momentum of the upper and lower states (assumed to be the same)
    B: scalar
        magnetic field strength
    line_strengths: None or np.array with the size [n, m]
        Line strength of the transitions between the upper and lower states
        If single L1-L2 lines are considered, we may leave it as 1.
    return_xr: returns xr.Dataset if True

    Returns
    -------

    '''
    energy_upper, J_upper, L_upper = np.broadcast_arrays(energy_upper, J_upper, L_upper)
    energy_lower, J_lower, L_lower = np.broadcast_arrays(energy_lower, J_lower, L_lower)
    
    Mmax = np.maximum(np.max(J_upper), np.max(J_lower))
    nM = int(2 * Mmax) + 1
    iM = np.arange(nM)
    # diagonalize upper
    eig_upper, eigv_upper = np.zeros((nM, len(J_upper))), np.zeros((nM, len(J_upper), len(J_upper)))
    for L in np.unique(L_upper):
        Lindex = np.arange(len(L_upper))[L_upper == L]
        Ms, eig, eigv = diagonalize(
            energy_upper[Lindex], J_upper[Lindex], L, S, B, Mmax
        )
        # substitute the eigenvalues / eigenvectors
        eig_upper[:, Lindex] = eig
        eigv_upper[(iM[:, np.newaxis, np.newaxis], Lindex[:, np.newaxis], Lindex)] = eigv 
    assert (eigv_upper > 0).any()  

    eig_lower, eigv_lower = np.zeros((nM, len(J_lower))), np.zeros((nM, len(J_lower), len(J_lower)))
    for L in np.unique(L_lower):
        Lindex = np.arange(len(L_lower))[L_lower == L]
        Ms, eig, eigv = diagonalize(
            energy_lower[Lindex], J_lower[Lindex], L, S, B, Mmax
        )
        eig_lower[:, Lindex] = eig
        eigv_lower[(iM[:, np.newaxis, np.newaxis], Lindex[:, np.newaxis], Lindex)] = eigv
    assert (eigv_lower > 0).any()  

    # 3 corresponds to delta M = -1, 0, 1
    intensity = np.zeros((3, nM, len(J_upper), len(J_lower)))  # intensity
    energy = np.full((3, nM, len(J_upper), len(J_lower)), fill_value=np.nan)  # transition energy

    for j, delta_M in enumerate([-1, 0, 1]):
        for i in range(nM):
            if i + j - 1 < 0 or i + j - 1 >= nM:
                continue
            M1 = Ms[i]
            M2 = Ms[i + j - 1]
            q = -delta_M

            if line_strengths is None:
                sign = np.where(L_upper[:, np.newaxis] == L_lower + 1, +1, np.where(L_upper[:, np.newaxis] == L_lower - 1, -1, 0))
                ls = sign * (
                    (-1)**(S + 1 + L_upper[:, np.newaxis] + J_lower) * 
                    np.sqrt((2 * J_upper[:, np.newaxis] + 1) * (2 * J_lower + 1)) *
                    py3nj.wigner6j(
                        2 * L_upper[:, np.newaxis], (2 * J_upper[:, np.newaxis]).astype(int), int(2 * S),
                        (2 * J_lower).astype(int), (2 * L_lower).astype(int), 2, 
                        ignore_invalid=True
                    )
                )
            else:
                ls = np.sqrt(line_strengths)

            strength = (
                (-1)**(J_upper[:, np.newaxis] - M1) *
                np.sqrt((2 * J_upper[:, np.newaxis] + 1) * (2 * J_lower + 1)) *
                py3nj.wigner3j(
                    (2 * J_upper[:, np.newaxis]).astype(int), 2, (2 * J_lower).astype(int),
                    -(2 * M1).astype(int), 2 * q, (2 * M2).astype(int), 
                    ignore_invalid=True
                ) * ls
            )
            # upper eigenvectors
            eig_u = eig_upper[i]
            eigv_u = eigv_upper[i]
            eig_l = eig_lower[i + j - 1]
            eigv_l = eigv_lower[i + j - 1]
            intensity[j, i] = (eigv_u.T @ strength @ eigv_l)**2
            energy[j, i] = eig_u[:, np.newaxis] - eig_l

    if return_xr:
        import xarray as xr
        return xr.Dataset({
            'intensity': (('deltaM', 'M', 'upper', 'lower'), intensity),
        }, coords={
            'deltaE': (('deltaM', 'M', 'upper', 'lower'), energy),
            'E_upper': (('M', 'upper'), eig_upper),
            'E_lower': (('M', 'lower'), eig_lower),
            'deltaM': [-1, 0, 1],
            'J_upper': ('upper', J_upper),
            'J_lower': ('lower', J_lower),
            'L_upper': ('upper', L_upper),
            'L_lower': ('lower', L_lower),
            'mixing_upper': (('M', 'upper', 'upper0'), eigv_upper),
            'mixing_lower': (('M', 'lower', 'lower0'), eigv_lower),
            'S': S, 'B': B, 'M': Ms
        })        
    return energy, intensity


def diagonalize(E, J, L, S, B, Mmax):
    r'''
    Compute the hamiltonian under the magnetic field

    Parameters
    ----------
    E: energy in eV
    J: total angular quantum number
    L: orbital angular quantum number
    S: spin angular quantum number
    B: magnetic field strength
    Mmax: M is computed from -M to M

    Returns
    -------
    M: angular momentum number
    eig: eigevalues [m, n]
    eigvec: eigenvectors [m, n, n]
    '''
    # size of hamiltonian
    n = len(J)
    H0 = np.eye(n) * E
    # for different M
    H = []
    Ms = np.arange(-Mmax, Mmax + 1)
    ML = np.arange(-L, L + 1)
    MS = np.arange(-S, S + 1)
    
    # Here, clebsch_gordan is a mutli-dimensional array:
    # 0th-axis: M, 1st-axis: MS, 2nd-axis: ML, 3rd-axis: [1], 4th-axis: J
    clebsch_gordan = py3nj.clebsch_gordan(
        2 * L, int(2 * S), (2 * J).astype(int), 
        np.expand_dims(2 * ML, axis=(1, 2)).astype(int), 
        np.expand_dims(2 * MS, axis=(1, 2, 3)).astype(int), 
        np.expand_dims(2 * Ms, axis=(1, 2, 3, 4)).astype(int), 
        ignore_invalid=True
    )
    Hpert = np.sum(
        clebsch_gordan * np.swapaxes(clebsch_gordan, -1, -2) * 
        (gL * np.expand_dims(ML, axis=(1, 2)) + gS * np.expand_dims(MS, axis=(1, 2, 3))), 
        axis=(1, 2)
    )
    assert (Hpert != 0).any()

    eig, eigv = np.linalg.eigh(H0 + -uB * B * Hpert)
    
    # This should be equivalent to the below, but in the vectorized manner
    '''
    ML = ML[:, np.newaxis, np.newaxis]
    MS = MS[:, np.newaxis]
    for M in Ms:
        # perturbed hamiltonian
        Hpert = np.zeros((n, n))
        for i1, j1 in enumerate(J):
            for i2, j2 in enumerate(J):
                if np.abs(M) > j1 or np.abs(M) > j2:
                    continue
                Hpert[i1, i2] = -uB * B * np.sum(
                    py3nj.clebsch_gordan(
                        2 * L, int(2 * S), (2 * j1).astype(int), 2 * ML, (2 * MS).astype(int), int(2 * M), 
                        ignore_invalid=True
                    ) * 
                    py3nj.clebsch_gordan(
                        2 * L, int(2 * S), (2 * j2).astype(int), 2 * ML, (2 * MS).astype(int), int(2 * M), 
                        ignore_invalid=True
                    ) * (gL * ML + gS * MS), axis=(0, 1)
                )
        H.append(H0 + Hpert)
    eig, eigv = np.linalg.eigh(np.stack(H, axis=0))
    '''
    return Ms, eig, eigv