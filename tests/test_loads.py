# -*- coding: utf-8 -*-

#    Copyright (C) 2017-2018 Mathew Topper
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np
import pytest

from dtocean_moorings.loads import (Loads,
                                    read_thrust_coefficient,
                                    get_rotor_load)


@pytest.fixture
def bathygrid():

    # 5 x 5 grid
    bathygrid = np.array([[0, 0, -10],
                          [0, 10, -10],
                          [0, 20, -10],
                          [0, 30, -10],
                          [0, 40, -10],
                          [10, 0, -20],
                          [10, 10, -20],
                          [10, 20, -20],
                          [10, 30, -20],
                          [10, 40, -20],
                          [20, 0, -30],
                          [20, 10, -30],
                          [20, 20, -30],
                          [20, 30, -30],
                          [20, 40, -30],
                          [30, 0, -40],
                          [30, 10, -40],
                          [30, 20, -40],
                          [30, 30, -40],
                          [30, 40, -40],
                          [40, 0, -50],
                          [40, 10, -50],
                          [40, 20, -50],
                          [40, 30, -50],
                          [40, 40, -50]])
    
    return bathygrid

def test_set_fairloc(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.fairloc = np.array([[0.0, 1.0, -1.5],
                                  [1.0, 0.0, -1.5],
                                  [0.0, -1.0, -1.5],
                                  [-1.0, 0.0, -1.5]])
    
    test = Loads(variables)
    test._set_fairloc()
    
    assert np.isclose(test.fairloc, variables.fairloc).all()
    assert np.isclose(test.fairlocglob, variables.fairloc).all()


# def test_set_fairloc_assert(mocker):
    
    # # Mock the variables class
    # variables = mocker.Mock()
    # variables.fairloc = np.array([[0.0, 1.0, -1.5],
                                  # [1.0, 0.0, -1.5],
                                  # [0.0, -1.0, -1.5],
                                  # [-1.0, 0.0, -1.5]])
    
    # test = Loads(variables)
    
    # with pytest.raises(AssertionError):
        # test._set_fairloc(3)


@pytest.mark.parametrize("systype", [
        'wavefloat', 
        'tidefloat',
        'wavefixed',
        'tidefixed',
        'substation'])
def test_get_foundation_quantity(systype, mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    
    foundloc = np.array([[0.0, 13.333, 0],
                         [7.5, -6.667, 0],
                         [-7.5, -6.667, 0]])
    
    test = Loads(variables)
    result = test._get_foundation_quantity(systype, foundloc)
    
    assert result == 3


def test_get_foundation_quantity_bad_systype(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    
    foundloc = np.array([[0.0, 13.333, 0],
                         [7.5, -6.667, 0],
                         [-7.5, -6.667, 0]])
    
    test = Loads(variables)
    
    with pytest.raises(ValueError):
        test._get_foundation_quantity("OWC", foundloc)


def test_get_foundation_quantity_foundradnew(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    
    foundloc = np.array([[0.0, 13.333, 0],
                         [7.5, -6.667, 0],
                         [-7.5, -6.667, 0]])
    
    test = Loads(variables)
    test.foundradnew = 1.
    test.numlines = 1
    
    result = test._get_foundation_quantity('wavefloat', foundloc)
    
    assert result == 1
    
    
def test_get_minimum_distance(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.sysorig = {'device001': [0, 0, 0.0],
                         'device002': [10, 0, 0.0],
                         'device003': [20, 20, 0.0]}
    
    test = Loads(variables)
    result = test._get_minimum_distance([0, 0, 0.0])
    
    assert np.isclose(result, 10)
    

def test_get_minimum_distance_single(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.sysorig = {'device001': [0, 0, 0.0]}
    
    test = Loads(variables)
    result = test._get_minimum_distance([0, 0, 0.0])
    
    assert np.isclose(result, 10000.0)
    

def test_get_maximum_displacement(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.maxdisp = [0, 0, 0]
    variables.sysdraft = 10.
    
    test = Loads(variables)
    result = test._get_maximum_displacement()
    
    assert np.isclose(result, [0, 0, 0]).all()
    

def test_get_maximum_displacement_none(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.maxdisp = None
    
    test = Loads(variables)
    result = test._get_maximum_displacement()
    
    assert np.isclose(result, [1000.0, 1000.0, 1000.0]).all()
    

def test_get_foundation_locations_foundloc(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundloc = np.array([[0.0, 13.333, 0],
                                   [7.5, -6.667, 0],
                                   [-7.5, -6.667, 0]])
    
    lineangs = [math.pi / 3., math.pi / 3, math.pi / 3]
    
    test = Loads(variables)
    result = test._get_foundation_locations(lineangs)
    
    assert np.isclose(result, variables.foundloc).all()
    
    
def test_get_foundation_locations_prefootrad(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundloc = None
    variables.prefootrad = 10.
    
    lineangs = [0., 2 * math.pi / 3, 4 *  math.pi / 3]
    
    test = Loads(variables)
    result = test._get_foundation_locations(lineangs)
    
    expected = np.array([[0., 10., 0.],
                         [5 * math.sqrt(3), -5., 0.],
                         [-5 * math.sqrt(3), -5., 0.]])
    
    assert np.isclose(result, expected).all()
    
    
def test_get_foundation_locations_meandepth(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundloc = None
    variables.prefootrad = None
    variables.bathygrid = bathygrid
    
    lineangs = [0., 2 * math.pi / 3, 4 *  math.pi / 3]
    
    test = Loads(variables)
    result = test._get_foundation_locations(lineangs)
    
    expected = np.array([[0., 240., 0.],
                         [120 * math.sqrt(3), -120., 0.],
                         [-120 * math.sqrt(3), -120., 0.]])
        
    assert np.isclose(result, expected).all()


def test_get_foundation_locations_quanfound(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.foundloc = None
    
    lineangs = [0., 2 * math.pi / 3, 4 *  math.pi / 3]
    
    test = Loads(variables)
    
    with pytest.raises(AssertionError):
        test._get_foundation_locations(lineangs, 4)
        
        
def test_get_closest_grid_point(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (21, 21)
    
    test = Loads(variables)
    index, point = test._get_closest_grid_point(sysorig)
    
    assert index == 12
    assert np.isclose(point, bathygrid[index, :]).all()


def test_get_closest_grid_point_local(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (21, 21)
    local = (8, 8)
    
    test = Loads(variables)
    index, point = test._get_closest_grid_point(sysorig, local)
    
    assert index == 18
    assert np.isclose(point, bathygrid[index, :]).all()


def test_get_closest_grid_point_fail(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (50, 50)
    
    test = Loads(variables)
    
    with pytest.raises(RuntimeError):
        test._get_closest_grid_point(sysorig)


def test_get_neighbours(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (21, 21)
    closest_point = (20, 20, -20)
    
    test = Loads(variables)
    indexes, points = test._get_neighbours(sysorig,
                                           closest_point)
    
    assert set(indexes) == set([12, 13, 17, 18])
    assert np.isclose(points, bathygrid[indexes, :]).all()


def test_get_neighbours_grid(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (20, 21)
    closest_point = (20, 20, -20)
    
    test = Loads(variables)
    indexes, points = test._get_neighbours(sysorig,
                                           closest_point)
        
    assert set(indexes) == set([6, 8, 16, 18])
    assert np.isclose(points, bathygrid[indexes, :]).all()
    
    
def test_get_neighbours_local(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygriddeltax = 10.
    variables.bathygriddeltay = 10.
    variables.bathygrid = bathygrid
    
    sysorig = (21, 21)
    local = (8, 8)
    closest_point = (30, 30, -30)
    
    test = Loads(variables)
    indexes, points = test._get_neighbours(sysorig,
                                           closest_point,
                                           local)
            
    assert set(indexes) == set([12, 13, 17, 18])
    assert np.isclose(points, bathygrid[indexes, :]).all()
    
    
def test_get_depth(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygrid = bathygrid
    variables.wlevmax = 0.
    
    point = (21, 21)
    indexes = [12, 13, 17, 18]
    
    test = Loads(variables)
    result = test._get_depth(point, indexes)
    
    assert np.isclose(result, -31)
    
    
def test_get_depth_grid(mocker, bathygrid):
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.bathygrid = bathygrid
    variables.wlevmax = 0.
    
    point = (19, 20)
    indexes = [6, 8, 16, 18]
    
    test = Loads(variables)
    result = test._get_depth(point, indexes)
    
    assert np.isclose(result, -29)
    
    
def test_get_soil_type_depth_inf(mocker, bathygrid):
    
    soiltypgrid = [[0, 0, "hr", np.inf],
                   [0, 10, "hr", np.inf],
                   [0, 20, "hr", np.inf],
                   [0, 30, "hr", np.inf],
                   [0, 40, "hr", np.inf],
                   [10, 0, "hr", np.inf],
                   [10, 10, "hr", np.inf],
                   [10, 20, "hr", np.inf],
                   [10, 30, "hr", np.inf],
                   [10, 40, "hr", np.inf],
                   [20, 0, "hr", np.inf],
                   [20, 10, "hr", np.inf],
                   [20, 20, "hr", np.inf],
                   [20, 30, "hr", np.inf],
                   [20, 40, "hr", np.inf],
                   [30, 0, "hr", np.inf],
                   [30, 10, "hr", np.inf],
                   [30, 20, "hr", np.inf],
                   [30, 30, "hr", np.inf],
                   [30, 40, "hr", np.inf],
                   [40, 0, "hr", np.inf],
                   [40, 10, "hr", np.inf],
                   [40, 20, "hr", np.inf],
                   [40, 30, "hr", np.inf],
                   [40, 40, "hr", np.inf]]
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.soiltypgrid = soiltypgrid

    test = Loads(variables)
    result = test._get_soil_type_depth(10)
    
    assert result == ("hr", np.inf)
    

def test_get_soil_type_depth_skim(mocker, bathygrid):
    
    soiltypgrid = [[0, 0, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [0, 10, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [0, 20, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [0, 30, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [0, 40, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [10, 0, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [10, 10, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [10, 20, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [10, 30, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [10, 40, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [20, 0, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [20, 10, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [20, 20, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [20, 30, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [20, 40, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [30, 0, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [30, 10, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [30, 20, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [30, 30, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [30, 40, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [40, 0, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [40, 10, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [40, 20, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [40, 30, "ls", 0.2, "sc", 4., "hr", np.inf],
                   [40, 40, "ls", 0.2, "sc", 4., "hr", np.inf]]
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.soiltypgrid = soiltypgrid

    test = Loads(variables)
    result = test._get_soil_type_depth(10)
    
    assert result == ("hr", np.inf)


def test_get_soil_type_depth_sediment(mocker, bathygrid):
    
    soiltypgrid = [[0, 0, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [0, 10, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [0, 20, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [0, 30, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [0, 40, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [10, 0, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [10, 10, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [10, 20, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [10, 30, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [10, 40, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [20, 0, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [20, 10, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [20, 20, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [20, 30, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [20, 40, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [30, 0, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [30, 10, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [30, 20, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [30, 30, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [30, 40, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [40, 0, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [40, 10, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [40, 20, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [40, 30, "ls", 0.2, "sc", 6., "hr", np.inf],
                   [40, 40, "ls", 0.2, "sc", 6., "hr", np.inf]]
    
    # Mock the variables class
    variables = mocker.Mock()
    variables.soiltypgrid = soiltypgrid

    test = Loads(variables)
    result = test._get_soil_type_depth(10)
    
    assert result == ("sc", 6.2)


@pytest.mark.parametrize("soiltype, expected", [
        ('ls', 'cohesionless'),
        ('ms', 'cohesionless'),
        ('ds', 'cohesionless'),
        ('vsc', 'cohesive'),
        ('sc', 'cohesive'),
        ('fc', 'cohesive'),
        ('stc', 'cohesive'),
        ('hgt', 'other'),
        ('cm', 'other'),
        ('src', 'other'),
        ('hr', 'other'),
        ('gc', 'other')])
def test_get_soil_group(mocker, soiltype, expected):
    
    # Mock the variables class
    variables = mocker.Mock()

    test = Loads(variables)
    result = test._get_soil_group(soiltype)
    
    assert result == expected


def test_get_soil_group_fail(mocker):
    
    # Mock the variables class
    variables = mocker.Mock()

    test = Loads(variables)
    
    with pytest.raises(ValueError):
        test._get_soil_group("bob")


@pytest.mark.parametrize("currentvelhub, expected", [
        (0.5, 0.5),
        (-1, 0),
        (2, 1)])
def test_read_thrust_coefficient(currentvelhub, expected):
    
    thrustcurv = np.array([[0, 0], [1, 1]])
    test = read_thrust_coefficient(thrustcurv, currentvelhub)
    
    assert np.isclose(test, expected)


@pytest.mark.parametrize("currentangattk, use_max_thrust, expected", [
        (0, False, (-1, 0, 0)),
        (90, False, (0, 1, 0)),
        (180, False, (1, 0, 0)),
        (270, False, (0, -1, 0)),
        (0, True, (-2, 0, 0)),])
def test_get_rotor_load(currentangattk, use_max_thrust, expected):
    
    thrustcurv = np.array([[0, 0], [1, 1]])
    test = get_rotor_load(thrustcurv,
                          0.5,
                          16,
                          currentangattk,
                          1,
                          use_max_thrust)
    
    assert np.isclose(test, expected).all()
