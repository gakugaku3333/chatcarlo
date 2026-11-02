"""P7 scoring metadata and the explicitly registered P3b-3 crop."""
import numpy as np

SIZES=(120,300,120,*([160]*6),*([320]*6))
VOLUMES=np.array([1.,.4,.25,*([.5]*6),*([.25]*6)])


def crop_p6(values):
    values=np.asarray(values)
    if values.shape[-1]!=3420:raise ValueError('P7 layout requires 3420 bins')
    pieces=[values[...,:540]]
    for j in range(6):pieces.append(values[...,540+j*160+44:540+j*160+116])
    for j in range(6):pieces.append(values[...,1500+j*320+88:1500+j*320+232])
    return np.concatenate(pieces,axis=-1)


def masses():
    return np.repeat(VOLUMES,SIZES)  # water, 1 g/cm^3


def metadata(n):
    return dict(physics_set='therapy-electron-v1',layout='p7',lateral_extent_cm=20,
        arrays=dict(absorbed_dose_Gy=dict(quantity='absorbed-dose',unit='Gy/history',normalization=f'divided by {n} histories'),
                    field_MeV=dict(quantity='deposited-energy',unit='MeV',normalization='raw sum')))


def dose(field,n):
    if type(n) is not int or n<=0:raise ValueError('positive histories required')
    return np.asarray(field)*1.602176634e-10/masses()/n
