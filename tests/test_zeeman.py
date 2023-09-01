import pytest
import numpy as np
from lineshapes import zeeman
import pyspectra


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
        L_upper, L_lower, S=1/2, B=0, return_xr=True
    )
    # eigen values should be the same
    assert (data['E_upper'] == data['E_upper'][0]).all()
    assert (data['E_upper'] == upper['energy'].values).all()
    assert (data['E_lower'] == data['E_lower'][0]).all()
    assert (data['E_lower'] == lower['energy'].values).all()
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 301)

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)
    
    data = zeeman.LS(
        upper['energy'].values, lower['energy'].values, 
        upper['J'].values / 2, lower['J'] / 2, 
        L_upper, L_lower, S=1/2, B=0, line_strengths=line_strength,
        return_xr=True
    )

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)

    # with B
    data = zeeman.LS(
        upper['energy'].values, lower['energy'].values, 
        upper['J'].values / 2, lower['J'] / 2, 
        L_upper, L_lower, S=1/2, B=10.0, line_strengths=line_strength,
        return_xr=True
    )
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 301)
    assert (data['E_upper'] != upper['energy'].values).any()
    assert (data['E_lower'] != lower['energy'].values).any()

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])
    assert np.allclose(np.sum(hist[0]), np.sum(hist, axis=1))
    

def test_zeeman_hydrogen():
    H = pyspectra.data.atom_levels('H', 1)
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
        S=1/2, B=B, return_xr=True
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
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)
    
    data = zeeman.LS(
        upper_energy, lower_energy,
        upper_J, lower_J, upper_L, lower_L,
        S=1/2, B=B, line_strengths=line_strengths, return_xr=True
    )
    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])
    assert np.allclose(hist[0], hist)

    # with B
    B = 3.0
    data = zeeman.LS(
        upper_energy, lower_energy,
        upper_J, lower_J, upper_L, lower_L,
        S=1/2, B=B, line_strengths=line_strengths, return_xr=True
    )
    data['deltaE'] = pyspectra.refractive_index.vacuum_to_air(pyspectra.units.eV_to_nm(data['deltaE']))
    bins = np.linspace(data['deltaE'].min(), data['deltaE'].max(), 101)
    #assert (data['E_upper'] != upper_energy).any()
    #assert (data['E_lower'] != lower_energy).any()

    # all the intensities should be the same for all the q-components
    hist = []
    for i in range(3):
        da = data.isel(deltaM=i)
        hist.append(np.histogram(da['deltaE'].values.ravel(), weights=da['intensity'].values.ravel(), bins=bins)[0])

    assert np.allclose(np.sum(hist[0]), np.sum(hist, axis=1))
    
