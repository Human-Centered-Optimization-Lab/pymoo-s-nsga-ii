import pytest

from pymoo.algorithms.moo.snsga2 import VSSPS

import numpy as np

test_striping_io = [
    # prob, n_samples
    (type('obj', (object,), {'n_var' : 100}), 100)
        ]

@pytest.mark.parametrize('prob, n_samples', test_striping_io)
def test_striping(prob, n_samples): 
   
    sampler = VSSPS()
    
    sampler._do(prob, n_samples)

    assert True


