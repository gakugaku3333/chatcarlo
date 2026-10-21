"""Ordered aggregation of verified native batches, shared by all G1 drivers."""
import numpy as np


def aggregate(batches, names, detector=None, roi=None, track_uncertainty=True):
    from ..transport import TransportResult
    from ..detector import DetectorTally
    from ..tally import ScalarMoments
    tally = DetectorTally(detector, track_uncertainty, roi) if detector is not None else None
    moments = ScalarMoments() if track_uncertainty else None
    deposits = {name: 0. for name in names}
    endings = np.zeros(3)
    scatter = fluor = 0.
    histories = 0
    for n, r in batches:
        histories += n
        batch = {name: float(v) for name, v in zip(names, r['deposits'])}
        for name, value in batch.items(): deposits[name] += value
        if moments is not None: moments.add_batch(batch, n)
        endings += r['endings']
        scatter += r['scatter_sum']
        fluor += r['n_fluorescence']
        if tally is not None:
            tally.category_fluence += r['images'][:4]
            tally.photon_count += r['images'][4]
            tally.energy_sum_keV += r['images'][5]
            tally.energy_sum2_keV2 += r['images'][6]
            tally.end_batch(n)
    if histories <= 0: raise ValueError('no histories to aggregate')
    return TransportResult(histories, {k: v/1000 for k, v in deposits.items()},
        float(endings[0]/histories), float(endings[1]/histories), float(scatter/histories),
        n_fluorescence=int(fluor), n_batches=moments.n_batches if moments is not None else 0,
        energy_deposited_sem_MeV={k: v/1000 for k, v in moments.standard_errors().items()} if moments is not None else {},
        energy_deposited_rel_err=moments.relative_errors() if moments is not None else {}, detector=tally)
