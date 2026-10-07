"""Measure existing production photon kernel at 60 keV. No MV transport is claimed.
Count actual positive flight segments with its existing instrumented kernel, and
verify identical transport totals against the untallied kernel for each fixed seed.
"""
from pathlib import Path
import os
BASE=Path(__file__).resolve().parent
os.environ['NUMBA_CACHE_DIR']=str(BASE/'numba_cache')
os.environ['NUMBA_NUM_THREADS']='1'
import sys,time,json,hashlib,platform
sys.path.insert(0,str(BASE.parents[3]))
import numpy as np
import numba
from chatcarlo import kernel as k

def counted(t,g,n,seed):
 seeds,offsets,counts=k._chunk_plan(n,1,seed);cap=n*64
 seg_o=np.zeros((1,cap,3));seg_d=np.zeros_like(seg_o)
 seg_ds=np.zeros((1,cap));seg_e=np.zeros_like(seg_ds);seg_mat=np.zeros((1,cap),dtype=np.int64)
 args=(n,1,seeds,offsets,counts,60.,0.,0.,-.0001,0.,0.,1.,g.n_boxes,g.box_center,g.box_half,g.box_material,g.background_material,g.world_center,g.world_half,t.n_elem,t.zs,t.fracs,t.log_e,t.step,t.n_grid,t.density_g_cm3,t.photo,t.compt,t.rayl,t.incoh_q,t.incoh_s,t.rayl_x,t.rayl_a,t.k_edge,t.k_omega,t.k_frac,t.k_line_e,t.k_line_p,t.n_lines,t.zs.shape[1],False,seg_o,seg_d,seg_ds,seg_e,seg_mat,cap)
 start=time.perf_counter();r=k._run_batch_scalar_tally(*args);elapsed=time.perf_counter()-start
 assert not np.any(r[7]);return r,elapsed,int(r[6].sum()),sum(a.nbytes for a in [seg_o,seg_d,seg_ds,seg_e,seg_mat])

def main():
 # Only use this T4 measurement after the MV reference energy balance succeeds.
 check=json.loads((BASE/'egs5_check.json').read_text());assert check['J3']=='pass'
 t=k.bake_scene_materials(['water','air']);t.density_g_cm3[t.code('air')]=1e-30
 g=k.bake_box_scene([{'center':(0,0,15),'size_cm':(30,30,30),'material':'water'}],'air',t,bbox_margin_cm=1.)
 k.run_batch(t,g,60.,(0,0,-.0001),(0,0,1),100,20261007,fluorescence_enabled=False)
 counted(t,g,100,20261007)
 rows=[]
 for n in [20000,100000]:
  for repeat in range(3):
   seed=20261007+repeat
   start=time.perf_counter();r=k.run_batch(t,g,60.,(0,0,-.0001),(0,0,1),n,seed,fluorescence_enabled=False);elapsed=time.perf_counter()-start
   c,ct,segments,memory=counted(t,g,n,seed)
   assert np.array_equal(r.n_scatter,c[0]) and np.array_equal(r.energy_deposited,c[4])
   rows.append({'histories':n,'seed':seed,'transport_seconds':elapsed,'recorded_transport_seconds':ct,'positive_flight_segments':segments,'segments_per_history':segments/n,'seconds_per_segment_untallied':elapsed/segments,'seconds_per_segment_recorded':ct/segments,'segment_buffer_allocated_bytes':memory})
 result={'rows':rows,'basis':'60 keV diagnostic kernel, one thread, JIT warmup excluded; timing includes existing material/geometry/boundary handling. MV electron, pair/positron and finer scoring costs are not measured here.','python':sys.version,'platform':platform.platform(),'numba':numba.__version__,'numpy':np.__version__,'kernel_sha256':hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest()}
 (BASE/'kernel_measurement.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
