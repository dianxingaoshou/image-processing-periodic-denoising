import numpy as np
import pytest

import nine_chirp


@pytest.mark.parametrize("transpose", [False, True])
def test_training_only_recovers_center_and_mixed_curvature_on_a_rectangle(transpose):
    h, w = 83, 109
    y, x = np.indices((h, w), dtype=float)
    train = ((4*y//h + 4*x//w) % 2) == 0
    y, x = (y-(h-1)/2)/h, (x-(w-1)/2)/w
    params = np.array([12., -17., -8., 4., 12.])
    cycles = params[0]*y + params[1]*x + .5*params[2]*y*y + params[3]*y*x + .5*params[4]*x*x
    image = np.cos(2*np.pi*cycles+.21)
    if transpose:
        image, train = image.T, train.T
        params = params[[1,0,4,3,2]] * -1
    first = nine_chirp.propose(image, (.025,.45), train)
    assert any(np.max(abs(seed-params)) < 1e-9 for seed in first)
    image[~train] = np.nan
    np.testing.assert_array_equal(nine_chirp.propose(image,(.025,.45),train), first)


def test_no_seeds_without_a_varying_fitting_signal():
    image = np.ones((64, 64))
    assert not nine_chirp.propose(image,(.025,.45),image==0)
    assert not nine_chirp.propose(image,(.025,.45),image==1)
