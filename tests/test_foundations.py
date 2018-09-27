# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 11:41:23 2017

@author: mtopper
"""

import math

import pytest
import numpy as np

from dtocean_moorings.foundations import Found


def test_get_sliding_resistance(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundsf = 1.
    
    test = Found(variables)
    test.soilfric = 0.3
    
    slope = math.pi / 36
    load = 100
    
    result = test._get_sliding_resistance(slope, load)

    assert np.isclose(result, 482.913813486)


def test_get_sliding_resistance_fail(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundsf = 1.
    
    test = Found(variables)
    test.soilfric = 0.3
    
    slope = 0.3
    load = 100
    
    result = test._get_sliding_resistance(slope, load)

    assert not result


@pytest.mark.parametrize("prefound_in, prefound_out, all_has_gravity",
    [("gravity", "gravity", True),
     ("uniarygravity", "gravity", True),
     ([], None, False)
    ])
def test_foundsel_float(mocker, prefound_in, prefound_out, all_has_gravity):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.prefound = prefound_in
    
    test = Found(variables)
    
    # Fake some attributes
    test.quanfound = 5
    test.soildep = [0., 0.5, 2., 7, 26.]
    test.soiltyp = ['ls', 'ls', 'vsc', 'sc', 'hr']
    test.loadmag = [440e3, 440e3, 445e3, 4450e3, 4450e3]
    test.loaddir = [-1., -1., 0, 1., 1.]
    test.seabedslpk = 'steep'
    
    test.foundsel("wavefloat")
    
    foundations = [[y[0] for y in x] for x in test.possfoundtyp]
    has_gravity = ["gravity" in y for y in foundations]
    
    assert len(test.possfoundtyp) == test.quanfound
    assert all(has_gravity) is all_has_gravity
    assert test.prefound == prefound_out


def test_foundsel_fixed(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.prefound = 'drag'
    
    test = Found(variables)
    
    # Fake some attributes
    test.quanfound = 5
    test.soildep = [0., 0.5, 2., 7, 26.]
    test.soiltyp = ['ls', 'ls', 'vsc', 'sc', 'hr']
    test.loadmag = [440e3, 440e3, 445e3, 4450e3, 4450e3]
    test.loaddir = [-1., -1., 0, 1., 1.]
    test.seabedslpk = 'steep'
    
    test.foundsel("wavefixed")
    
    foundations = [[y[0] for y in x] for x in test.possfoundtyp]
    has_drag = ["drag" in y for y in foundations]
    
    assert len(test.possfoundtyp) == test.quanfound
    assert not all(has_drag)
