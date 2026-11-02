"""Independent bin-index oracle and units; no statistical judgement is run."""
import numpy as np
import pytest
from chatcarlo.gpu.therapy_layout import crop_p6,dose,masses


def test_crop_every_element():
    data=np.arange(3*3420).reshape(3,3420)
    # Independent expected indices enumerated by physical bin centres.
    expected=[];offset=0
    for size,width in [(120,.25),(300,.1),(120,.25)]+[(160,.25)]*6+[(320,.125)]*6:
        for i in range(size):
            if offset<540 or -9< -20+(i+.5)*width <9:expected.append(offset+i)
        offset+=size
    assert len(expected)==1836
    np.testing.assert_array_equal(crop_p6(data),data[:,expected])
    with pytest.raises(ValueError):crop_p6(data[:,:100])


def test_artificial_regression_embedding():
    # Cache-backed saved-entry regression is deliberately left for Claude.
    p6=np.arange(2*1836,dtype=float).reshape(2,1836)+.125
    p7=np.full((2,3420),-999.)
    p7[:,:540]=p6[:,:540];a=540
    for base,size,left,count in [(540,160,44,72),(1500,320,88,144)]:
        for j in range(6):
            p7[:,base+j*size+left:base+j*size+left+count]=p6[:,a:a+count];a+=count
    np.testing.assert_array_equal(crop_p6(p7),p6)


def test_dose_units():
    field=masses()*100
    np.testing.assert_allclose(dose(field,100),np.full(3420,1.602176634e-10),rtol=1e-15)
