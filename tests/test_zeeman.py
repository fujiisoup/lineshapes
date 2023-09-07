import os
import pytest
import numpy as np
from lineshapes import zeeman
import xarray as xr
import pyspectra


THIS_DIR = os.path.dirname(__file__)


def allclose_except_nans(da, db, *args, **kwargs):
    da = np.where(np.isnan(da), db, da)
    db = np.where(np.isnan(db), da, db)
    da = np.where(np.isnan(da), 0, da)
    db = np.where(np.isnan(db), 0, db)
    return np.allclose(da, db, *args, **kwargs)    


def test_ignore_LS_selectionrules():
    energy_upper = [19.30534490, 19.261084163, 19.222902151]  # 4p 4P
    L_upper = [1, 1, 1]
    J_upper = [1/2, 3/2, 5/2]
    energy_lower = [16.81247225, 16.74852891, 16.64385439]  # 4s 4P
    L_lower = [1, 1, 1]
    J_lower = [1/2, 3/2, 5/2]
    Spq = np.array([
        # from 4p(1/2)   4p(3/2)    4p(5/2)
        [ 1.2e+00, 5.72e+00,      0.0],  # to 4s (1/2)
        [9.56e+00, 3.42e+00, 5.63e+00],  # to 4s (3/2)
        [     0.0, 1.22e+01, 2.57e+01],  # to 4s (5/2)
    ]).T

    zeeman_Ar = zeeman.LS(
        energy_upper, energy_lower, 
        J_upper=J_upper, J_lower=J_lower, 
        L_upper=L_upper, L_lower=L_lower, S_upper=3/2, S_lower=3/2, B=1, 
        line_strengths_LJ=Spq,
        ignore_LS_selectionrules=True,
        return_xr=True
    )

    wl = 473.58
    bins = np.linspace(wl - 0.1, wl + 0.1, 51)
    bins_center = (bins[:-1] + bins[1:]) / 2
    
    pi = np.histogram(
        pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(zeeman_Ar.sel(deltaM=0)['deltaE'])).values.ravel(),
        weights=0.1 * zeeman_Ar['strength'].sel(deltaM=0).values.ravel(), bins=bins
    )[0]
    sigma = np.histogram(
        pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(zeeman_Ar.sel(deltaM=[-1, 1])['deltaE'])).values.ravel(),
        weights=0.1 * zeeman_Ar['strength'].sel(deltaM=[-1, 1]).values.ravel(), bins=bins
    )[0] / 2
    total = np.sum(pi + sigma)
    assert np.sum(pi / total > 1e-4) <= 2
    assert np.sum(sigma / total  > 1e-4) <= 4
    assert np.sum(sigma[:25] / total > 1e-4) <= 2
    

def test_hydrogen():
    energy_upper = [
        12.0874936591, 12.0875071004,  # 3p (1/2, 3/2)
        12.0874949611,  # 2s (1/2)
        12.0875070783, 12.0875115582,  # 3d (3/2, 5/2)
    ]
    L_upper = [1, 1, 0, 2, 2]
    J_upper = [1/2, 3/2, 1/2, 3/2, 5/2]
    energy_lower = [
        10.19880615024, 10.19885151459,  # 2p (1/2, 3/2)
        10.19881052514816  # 2s (1/2)
    ]
    L_lower = [1, 1, 0]
    J_lower = [1/2, 3/2, 1/2]
    Spq = np.array([
        # from 3p(1/2)   3p(3/2)    3s(1/2)      3d(3/2)    3d(5/2)
        [       0.0,        0.0, 5.8769e-01, 3.0089e+01,        0.0],   # to 2p (1/2)
        [       0.0,        0.0, 1.1756e+00, 6.0180e+00, 5.4162e+01], # to 2p (3/2)
        [6.2687e+00, 1.2537e+01,        0.0,        0.0,        0.0],  # to 2s (1/2)
    ]).T

    zeeman_H = zeeman.LS(
        energy_upper, energy_lower, 
        J_upper=J_upper, J_lower=J_lower, 
        L_upper=L_upper, L_lower=L_lower, 
        S_upper=1/2, S_lower=1/2, 
        B=7, 
        line_strengths_LJ=Spq,
        return_xr=True
    )
    
    bins = np.linspace(1.887, 1.891, num=31)
    # pi component should be a single line
    pi = np.histogram(
        zeeman_H['deltaE'].sel(deltaM=0).values.ravel(), 
        weights=zeeman_H['A'].sel(deltaM=0).values.ravel(), 
        bins=bins
    )[0]
    pi = pi / np.sum(pi)

    # sigma component should be a doublet
    sigma = np.histogram(
        zeeman_H['deltaE'].sel(deltaM=[-1, 1]).values.ravel(), 
        weights=-zeeman_H['A'].sel(deltaM=[-1, 1]).values.ravel(), 
        bins=bins
    )[0]
    sigma = sigma / np.sum(sigma)
    import matplotlib.pyplot as plt
    assert np.sum(pi > 1e-3) < 2
    assert np.sum(sigma[:15] > 1e-3) < 2
    assert np.sum(sigma[15:] > 1e-3) < 3

    # without Spq, the result should be similar
    zeeman_H = zeeman.LS(
        energy_upper, energy_lower, 
        J_upper=J_upper, J_lower=J_lower, 
        L_upper=L_upper, L_lower=L_lower, 
        S_upper=1/2, S_lower=1/2, 
        B=7, 
        line_strengths_LJ=None,
        return_xr=True
    )
    pi2 = np.histogram(
        zeeman_H['deltaE'].sel(deltaM=0).values.ravel(), 
        weights=zeeman_H['A'].sel(deltaM=0).values.ravel(), 
        bins=bins
    )[0]
    pi2 = pi2 / np.sum(pi2)
    sigma2 = np.histogram(
        zeeman_H['deltaE'].sel(deltaM=[-1, 1]).values.ravel(), 
        weights=-zeeman_H['A'].sel(deltaM=[-1, 1]).values.ravel(), 
        bins=bins
    )[0]
    sigma2 = sigma2 / np.sum(sigma2)
    
    assert np.allclose(pi, pi2, atol=1e-3)
    assert np.allclose(sigma, sigma2, atol=1e-3)


def test_with_goto_singlet():
    # Helium n=4 singlet
    upper_energy = pyspectra.units.cm_to_eV(np.array([
        190940.330, 	# S=0, L=0, J=0
        191492.816,     # S=0, L=1, J=1
        191446.55901,   # S=0, L=2, J=2
        191451.9920,    # S=0, L=3, J=3
    ]))
    upper_L = np.array([0, 1, 2, 3])
    upper_J = np.array([0, 1, 2, 3])
    lower_energy = pyspectra.units.cm_to_eV(np.array([166277.542, 171135.00000]))
    lower_L = np.array([0, 1])
    lower_J = np.array([0, 1])
    line_strengths_L = np.array([
        [0.0, 0.662, 0.0, 0.0], 
        [0.416, 0.0, 5.95, 0.0],
    ]).T
    # Goto data				 
    goto = np.loadtxt(THIS_DIR + '/../goto/stark_zeeman.txt', skiprows=1, delimiter=',').T
    goto = xr.DataArray(
        goto[3:], dims=['xyz', 'index'], 
        coords={
            'wavelength': ('index', goto[2]), 
            'Bfield': ('index', goto[1]),
            'Efield': ('index', goto[0]),
            'xyz': ['x', 'y', 'z']
        }).set_index(index=['Efield', 'Bfield', 'wavelength'])
    goto = goto.sel(Efield=0).sel(Bfield=1.0)

    data = zeeman.LS(
        upper_energy, lower_energy, 
        upper_J, lower_J, upper_L, lower_L,
        S_upper=0, S_lower=0, B=goto['Bfield'].item(), line_strengths_L=line_strengths_L, 
        return_xr=True
    )
    
    # Compare the strength histogram
    for bins, dlam_correct in [
        (np.linspace(492.16, 492.22, num=301), 0.0006),
        (np.linspace(396.4, 396.5, num=301), 0.0005),
    ]:
        # pi component
        dM = 0
        actual_pi = np.histogram(
            pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(
                data['deltaE'].sel(deltaM=dM).values.ravel())
            ),
            weights=data['strength'].sel(deltaM=dM).values.ravel(), 
            bins=bins
        )[0]
        xyz = 'x'
        expected_pi = np.histogram(
            goto['wavelength'].values.ravel() + dlam_correct,
            weights=goto.sel(xyz=xyz).values.ravel(), 
            bins=bins
        )[0]
        #  sigma component
        dM = [-1, 1]
        actual_sigma = np.histogram(
            pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(
                data['deltaE'].sel(deltaM=dM).values.ravel())
            ),
            weights=data['strength'].sel(deltaM=dM).values.ravel(), 
            bins=bins
        )[0]
        xyz = ['y', 'z']
        expected_sigma = np.histogram(
            goto['wavelength'].values.ravel() + dlam_correct,
            weights=goto.sel(xyz=xyz).sum('xyz').values.ravel(), 
            bins=bins
        )[0]

        print(data)
        total_actual = np.nansum(actual_sigma) + np.nansum(actual_pi)
        total_expected = np.nansum(expected_sigma) + np.nansum(expected_pi)
        assert np.allclose(actual_pi / total_actual, expected_pi / total_expected)
        assert np.allclose(actual_sigma / total_actual, expected_sigma / total_expected)
        '''
        import matplotlib.pyplot as plt
        plt.plot(bins[:-1], actual_sigma / total_actual)
        plt.plot(bins[:-1], actual_pi / total_actual)
        plt.plot(bins[:-1], expected_sigma / total_expected, ls='--')
        plt.plot(bins[:-1], expected_pi / total_expected, ls='--')
        plt.show()
        '''


def test_with_goto_triplet():
    # Helium n=4 singlet
    upper_energy = pyspectra.units.cm_to_eV(np.array([
        190298.21651,   # S=1,L=0,J=1
        191217.2633,	# S=1,L=1,J=0
        191217.1530,    # S=1,L=1,J=1
        191217.1440,    # S=1,L=1,J=2
        191444.60399,   # S=1,L=2,J=1
        191444.58548,   # S=1,L=2,J=2
        191444.58427,   # S=1,L=2,J=3
        191451.9928,    # S=1,L=3,J=2
        191451.9855,    # S=1,L=3,J=3
        191451.9842,    # S=1,L=3,J=4
    ]))
    upper_L = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    upper_J = np.array([1, 0, 1, 2, 1, 2, 3, 2, 3, 4])
    lower_energy = pyspectra.units.cm_to_eV(np.array([
        159856.07760,   # S=1,L=0,J=1
        169087.934120,  # S=1,L=1,J=0
        169086.946208,  # S=1,L=1,J=1
        169086.869782,  # S=1,L=1,J=2
    ]))
    lower_L = np.array([0, 1, 1, 1])
    lower_J = np.array([1, 0, 1, 2])
    line_strengths_L = np.array([
        [   0.0, 0.2424, 0.2424, 0.2424,   0.0,   0.0,   0.0, 0.0, 0.0, 0.0], 
        [0.5484,    0.0,    0.0,    0.0, 5.538, 5.538, 5.538, 0.0, 0.0, 0.0],
        [0.5484,    0.0,    0.0,    0.0, 5.538, 5.538, 5.538, 0.0, 0.0, 0.0],
        [0.5484,    0.0,    0.0,    0.0, 5.538, 5.538, 5.538, 0.0, 0.0, 0.0],
    ]).T
    # Goto data				 
    goto = np.loadtxt(THIS_DIR + '/../goto/stark_zeeman.txt', skiprows=1, delimiter=',').T
    goto = xr.DataArray(
        goto[3:], dims=['xyz', 'index'], 
        coords={
            'wavelength': ('index', goto[2]), 
            'Bfield': ('index', goto[1]),
            'Efield': ('index', goto[0]),
            'xyz': ['x', 'y', 'z']
        }).set_index(index=['Efield', 'Bfield', 'wavelength'])
    goto = goto.sel(Efield=0).sel(Bfield=1.0)

    data = zeeman.LS(
        upper_energy, lower_energy, 
        upper_J, lower_J, upper_L, lower_L,
        S_upper=1, S_lower=1, 
        B=goto['Bfield'].item(), line_strengths_L=line_strengths_L, 
        return_xr=True
    )
    
    # Compare the strength histogram
    for bins, dlam_correct in [
        (np.linspace(471.2, 471.4, num=301), 0.00055),
        #(np.linspace(396.4, 396.5, num=301), 0.0005),
    ]:
        # pi component
        dM = 0
        actual_pi = np.histogram(
            pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(
                data['deltaE'].sel(deltaM=dM).values.ravel())
            ),
            weights=data['strength'].sel(deltaM=dM).values.ravel(), 
            bins=bins
        )[0]
        xyz = 'x'
        expected_pi = np.histogram(
            goto['wavelength'].values.ravel() + dlam_correct,
            weights=goto.sel(xyz=xyz).values.ravel(), 
            bins=bins
        )[0]
        #  sigma component
        dM = [-1, 1]
        actual_sigma = np.histogram(
            pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(
                data['deltaE'].sel(deltaM=dM).values.ravel())
            ),
            weights=data['strength'].sel(deltaM=dM).values.ravel(), 
            bins=bins
        )[0]
        xyz = ['y', 'z']
        expected_sigma = np.histogram(
            goto['wavelength'].values.ravel() + dlam_correct,
            weights=goto.sel(xyz=xyz).sum('xyz').values.ravel(), 
            bins=bins
        )[0]
        total_actual = np.sum(actual_sigma) + np.sum(actual_pi)
        total_expected = np.sum(expected_sigma) + np.sum(expected_pi)
        assert np.allclose(actual_pi / total_actual, expected_pi / total_expected, atol=1e-4, rtol=1e-3)
        assert np.allclose(actual_sigma / total_actual, expected_sigma / total_expected, atol=1e-4, rtol=1e-3)
        
        '''
        import matplotlib.pyplot as plt
        plt.plot(bins[:-1], -actual_sigma / total_actual, color='C0')
        plt.plot(bins[:-1], actual_pi / total_actual, color='C1')
        plt.plot(bins[:-1], -expected_sigma / total_expected, ls='--', color='C0')
        plt.plot(bins[:-1], expected_pi / total_expected, ls='--', color='C1')
        plt.show()
        '''
        

def test_zeeman_helium():
    he = pyspectra.data.atom_levels('He', 1)
    lower = he.isel(energy=he['Configuration'].isin(['3s', '3p', '3d'])).sortby('energy')
    upper = he.isel(energy=he['Configuration'].isin(['4s', '4p', '4d', '4f'])).sortby('energy')
    
    he = pyspectra.data.atom_lines('He', 1)
    he = he.isel(wavelength=he['conf_i'].isin(lower['Configuration']) * he['conf_k'].isin(upper['Configuration']))   
    line_strength = np.zeros((upper.sizes['energy'], lower.sizes['energy']))
    for k in range(he.sizes['wavelength']):
        i = lower.get_index('energy').get_indexer([he['Ei(eV)'][k]], method='nearest')
        j = upper.get_index('energy').get_indexer([he['Ek(eV)'][k]], method='nearest')
        line_strength[j, i] = he['fik'][k]
    
    L = 'SPDFGH'
    L_upper = [L.index(term.item()[-1]) for term in upper['Term']]
    L_lower = [L.index(term.item()[-1]) for term in lower['Term']]
    B = 0
    data = zeeman.LS(
        upper['energy'].values, lower['energy'].values, 
        upper['J'].values / 2, lower['J'] / 2, 
        L_upper, L_lower, S_upper=1/2, S_lower=1/2, B=0, return_xr=True
    )
    # eigen values should be the same
    assert allclose_except_nans(data['E_upper'], data['E_upper'][0])
    assert allclose_except_nans(data['E_upper'], upper['energy'])
    assert allclose_except_nans(data['E_lower'], data['E_lower'][0])
    assert allclose_except_nans(data['E_lower'], lower['energy'].values)
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 301)

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)
    
    data = zeeman.LS(
        upper['energy'].values, lower['energy'].values, 
        upper['J'].values / 2, lower['J'] / 2, 
        L_upper, L_lower, S_upper=1/2, S_lower=1/2, B=0, line_strengths_LJ=line_strength,
        return_xr=True
    )

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)

    # with B
    data = zeeman.LS(
        upper['energy'].values, lower['energy'].values, 
        upper['J'].values / 2, lower['J'] / 2, 
        L_upper, L_lower, S_upper=1/2, S_lower=1/2, B=10.0, line_strengths_LJ=line_strength,
        return_xr=True
    )
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 301)
    assert ~allclose_except_nans(data['E_upper'], upper['energy'].values)
    assert ~allclose_except_nans(data['E_lower'], lower['energy'].values)

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])
    assert np.allclose(np.sum(hist[0]), np.sum(hist, axis=1))
    

def _test_zeeman_hydrogen():
    lower_energy = [10.19880615024, 10.19885151459, 10.19881052514816]
    lower_J = [1/2, 3/2, 1/2]
    lower_L = [1, 1, 0]
    upper_energy = [12.0874936591, 12.0875071004, 12.0874949611, 12.0875070783, 12.0875115582]
    upper_J = [1/2, 3/2, 1/2, 3/2, 5/2]
    upper_L = [1, 1, 0, 2, 2]
    line_strengths = [
        [0, 0, 6.2687e+00], # p_1/2 ->
        [0, 0, 1.2537e+01], # p_3/2 ->
        [5.8769e-01, 1.1756e+00, 0], # S_1/2 ->
        [3.0089e+01, 6.0180e+00, 0], # D_3/2 ->
        [0, 5.4162e+01, 0], # D_5/2 ->
    ]
    
    B = 0.0
    data = zeeman.LS(
        upper_energy, lower_energy,
        upper_J, lower_J, upper_L, lower_L,
        S_upper=1/2, S_lower=1/2, B=B, return_xr=True
    )
    # eigen values should be the same
    assert (data['E_upper'] == data['E_upper'][0]).all()
    assert (data['E_upper'] == upper_energy).all()
    assert (data['E_lower'] == data['E_lower'][0]).all()
    assert (data['E_lower'] == lower_energy).all()
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 301)

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)
    
    data = zeeman.LS(
        upper_energy, lower_energy,
        upper_J, lower_J, upper_L, lower_L,
        S_upper=1/2, S_lower=1/2, B=B, line_strengths_LJ=line_strengths, return_xr=True
    )
    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)

    # with B
    B = 3.0
    data = zeeman.LS(
        upper_energy, lower_energy,
        upper_J, lower_J, upper_L, lower_L,
        S_upper=1/2, S_lower=1/2, B=B, line_strengths_LJ=line_strengths, return_xr=True
    )
    data['deltaE'] = pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(data['deltaE']))
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 101)
    #assert (data['E_upper'] != upper_energy).any()
    #assert (data['E_lower'] != lower_energy).any()

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['strength'].values.ravel(), bins=bins)[0])

    assert np.allclose(np.sum(hist[0]), np.sum(hist, axis=1))
    
