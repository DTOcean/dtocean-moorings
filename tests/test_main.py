# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:41:23 2018

@author: Mathew Topper
"""

from dtocean_moorings.main import Main


def test_Main_init(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    Main(variables)

    assert True


