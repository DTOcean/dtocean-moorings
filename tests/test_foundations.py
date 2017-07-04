# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 11:41:23 2017

@author: mtopper
"""

import math

import numpy as np
import pytest

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
    
    with pytest.raises(RuntimeError):
        test._get_sliding_resistance(slope, load)
