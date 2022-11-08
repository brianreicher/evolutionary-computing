import pytest
import pandas as pd
from src.assignta import overallocation, conflicts, undersupport, unwilling, unpreferred


@pytest.fixture
def tests():
    test1 = pd.read_csv('test/test1.csv', header=None).to_numpy()
    test2 = pd.read_csv('test/test2.csv', header=None).to_numpy()
    test3 = pd.read_csv('test/test3.csv', header=None).to_numpy()

    return test1, test2, test3


def test_overallocation(tests):
    test1, test2, test3 = tests

    assert overallocation(test1) == 37
    assert overallocation(test2) == 41
    assert overallocation(test3) == 23


def test_conflicts(tests):
    test1, test2, test3 = tests

    assert conflicts(test1) == 8
    assert conflicts(test2) == 5
    assert conflicts(test3) == 2


def test_undersupport(tests):
    test1, test2, test3 = tests

    assert undersupport(test1) == 1
    assert undersupport(test2) == 0
    assert undersupport(test3) == 7


def test_unwilling(tests):
    test1, test2, test3 = tests

    assert unwilling(test1) == 53
    assert unwilling(test2) == 58
    assert unwilling(test3) == 43


def test_unpreferred(tests):
    test1, test2, test3 = tests

    assert unpreferred(test1) == 15
    assert unpreferred(test2) == 19
    assert unpreferred(test3) == 10
