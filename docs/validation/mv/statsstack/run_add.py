"""Stack (P8e) EGS5 additional runs: the P8e production branch of run_p8e.run('egs') with only N, seed and folders changed.
The s7 replay and pilot prerequisites are not re-run (they were checked for the saved production)."""
import sys, json, time
from pathlib import Path
import numpy as np
from prototypes.mv_electron import run_p8e as p
from prototypes.mv_electron.run_p3a import load_spectrum
from prototypes.mv_electron.materials_p8a import OUT as P8A_OUT
root = Path('~/.cache/chatcarlo/statsstack').expanduser()
n, seed = int(sys.argv[1]), int(sys.argv[2]); folder = f'add_{seed}'
manifest = json.loads((P8A_OUT/'manifest.json').read_text())
definitions = [p.load_reference(manifest['materials'][m]['definition'], m) for m in p.MATERIALS]
densities = [d['density_g_cm3'] for d in definitions]
edges, _, cum = load_spectrum(); spectrum = p.spectrum_text(edges, cum); media_set = 4; layers = p.LAYERS
inp = p.pegs_input([dict(definitions[i], medium_name='P8E_'+p.MATERIALS[i].upper()) for i in p.media_indices(media_set)])
ref = json.loads((p.OUT/'egs_production.json').read_text())
assert ref['layers'] == json.loads(json.dumps(layers)) and ref['densities_g_cm3'] == densities and ref['media_set'] == media_set and ref['fluor'] == 1
prod = json.loads((Path.home()/'.cache/chatcarlo/p8e/egs/production/identity.json').read_text())
want = dict(sources={p.CODE.name: p.CODE.read_text()}, input=inp, config=p.config_text(60_000_000, 820001, layers, media_set=media_set),
            spectrum_sha256=p.hashlib.sha256(spectrum.encode()).hexdigest(), egs_inputs=json.loads(json.dumps(p.external_sources())))
diff = [k for k in prod if prod[k] != want.get(k)]
print('identity diff vs P8e production (config rebuilt with its N, seed):', diff, flush=True)
if diff: sys.exit('identity mismatch')
t = time.time()
validate = lambda text: p.parse_egs(text, n, seed, layers, densities, media_set=media_set)
text, identity_hash = p.execute(root/'work'/folder, inp, p.config_text(n, seed, layers, media_set=media_set), spectrum, validate)
dose, health = p.parse_egs(text, n, seed, layers, densities, media_set=media_set)
grid = p.egs_layout(layers, densities, media_set); batch_blocks, mean, sem = p.block_statistics(dose, grid)
out = root/'out'; out.mkdir(parents=True, exist_ok=True); artifact = out/f'egs_{folder}.npz'
np.savez(artifact, dose=dose, batch_blocks=batch_blocks, **grid)
p.write(out/f'egs_{folder}.json', dict(N=n, seed=seed, fluor=1, media_set=media_set, layers=layers, densities_g_cm3=densities,
    units='MeV cm2/g/history', identity_sha256=identity_hash, complete_sha256=p.sha(root/'work'/folder/'complete.json'),
    artifact_sha256=p.sha(artifact), block_mean=mean.tolist(), block_SEM=sem.tolist(), **health))
print('done', folder, round(time.time()-t, 1), 's', flush=True)
