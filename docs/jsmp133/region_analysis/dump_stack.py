"""Write the pooled stack PDD (EGS5 1e8, GPU 1e8; docs/validation/mv/statsstack/pool.py, unchanged) to pooled_stack.npz."""
import sys, numpy as np
sys.argv = ['pool.py']
src = open('docs/validation/mv/statsstack/pool.py').read().split("print('220 bins:'")[0]
exec(src)
np.savez('docs/jsmp133/region_analysis/pooled_stack.npz', gm=gm, gs=gs, em=em, es=es, layer=layer, edges=edges)
print('220 bins:', json.dumps(c))
