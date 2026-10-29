"""Untimed, same-input per-history replay required before a statistical verdict."""
import numpy as np


def diagnostic_health(measured, replay, energy):
    energy=np.asarray(energy,float);n=len(energy)
    nm=len(measured['deposits'])
    # Detector bounds have already been checked by Core.run. Here validate all
    # record values/counters, including artificial records supplied by tests.
    if (replay.get('d',np.empty(0)).shape!=(n,nm+2) or
            replay.get('i',np.empty(0)).shape!=(n,9)):
        raise ValueError('diagnostic record shape/history count')
    if (not all(np.isfinite(v).all() for v in replay.values()) or
            np.any(replay['d']<0) or np.any(replay['i'][:,8]) or
            np.any((replay['i'][:,3]<0)|(replay['i'][:,3]>2))):
        raise ValueError('invalid diagnostic record')
    if (measured['endings'].sum()!=n or replay['endings'].sum()!=n or
            not np.array_equal(np.bincount(replay['i'][:,3],minlength=3),replay['endings'])):
        raise ValueError('diagnostic history conservation')
    remaining=np.where(replay['i'][:,3]==0,0.,replay['d'][:,0])
    deposited=replay['d'][:,1]
    error=np.abs(deposited+remaining-energy)
    if np.any(error>1e-10*np.maximum(energy,1)):
        raise ValueError('diagnostic per-history energy balance')
    if (not np.allclose(replay['d'][:,2:].sum(axis=1),deposited,rtol=1e-10,atol=1e-8) or
            not np.allclose(replay['d'][:,2:].sum(axis=0),replay['deposits'],rtol=1e-10,atol=1e-8)):
        raise ValueError('diagnostic record deposit mismatch')
    for key in measured:
        if key in ('d','i','trace'):continue
        if key not in replay or not np.allclose(measured[key],replay[key],rtol=1e-10,atol=1e-8):
            raise ValueError('diagnostic replay aggregate mismatch: '+key)
    return dict(status='pass',histories=n,max_energy_error_keV=float(error.max(initial=0)),
                method='untimed same-input record replay')


def checked_run(driver, inputs, **kwargs):
    kwargs=dict(kwargs);kwargs.pop('record',None);kwargs.pop('observe',None)
    measured=driver.run(*inputs,**kwargs)
    replay=driver.run(*inputs,record=True,**kwargs)
    report=diagnostic_health(measured,replay,inputs[2])
    return measured,report
