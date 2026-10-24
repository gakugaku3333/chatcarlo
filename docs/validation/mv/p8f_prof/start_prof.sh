#!/bin/zsh
# Start session p8fprof on a T4, upload the needed P8f export files (water baseline in 50 MB parts) and launch vm_prof.py.
set -u
HERE=${0:A:h}; EXPORT=$HOME/.cache/chatcarlo/p8f_prof/p8f/export; PARTS=$HOME/.cache/chatcarlo/p8f_prof/parts
mkdir -p $PARTS
cd /tmp && colab new -s p8fprof --gpu T4 | tail -1
printf 'import os\nfor d in ["/content/p8f","/content/parts"]:os.makedirs(d,exist_ok=True)\n' | colab exec -s p8fprof >/dev/null
cd $EXPORT
for f in water_bundle.npz lung_bundle.npz bone_bundle.npz lead_bundle.npz water_fluorescence.npz lung_fluorescence.npz bone_fluorescence.npz lead_fluorescence.npz bundles.json p8f_inputs.npz p8f_inputs.json p8f_source.zip p8f_manifest.json; do
  colab upload -s p8fprof "$f" "/content/p8f/$f" >/dev/null 2>&1 || colab upload -s p8fprof "$f" "/content/p8f/$f" >/dev/null 2>&1 || echo FAIL $f
done
[ -n "$(ls $PARTS)" ] || split -b 50m p8f_water_baseline.npz $PARTS/p8f_water_baseline.npz.part.
cd $PARTS && for p in *; do colab upload -s p8fprof "$p" "/content/parts/$p" >/dev/null 2>&1 || echo FAIL $p; done
colab upload -s p8fprof $HERE/vm_prof.py /content/vm_prof.py >/dev/null 2>&1 || echo FAIL vm_prof
colab upload -s p8fprof $HERE/prof_target.py /content/prof_target.py >/dev/null 2>&1 || echo FAIL target
colab exec -s p8fprof -f $HERE/assemble_prof.py | tail -3
