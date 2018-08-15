import pytest

from asi.helpers.annealing_schedules import get_linear, get_reciprocal


def test_get_reciprocal():
    reciprocal = get_reciprocal(0.01)
    assert reciprocal(0) == 1.0
    assert reciprocal(100) == 0.5
    assert reciprocal(200) == pytest.approx(1 / 3)


def test_get_linear():
    linear = get_linear(steps=1000, start_value=0.6, final_value=0.1)
    assert linear(0) == 0.6
    assert linear(500) == 0.35
    assert linear(1000) == 0.1
    assert linear(10000) == 0.1
