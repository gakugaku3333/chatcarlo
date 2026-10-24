#!/bin/zsh
# Session p8fline on a T4: upload the P8f material files and launch vm_line.py (launch-size test + ncu line map).
set -u
HERE=${0:A:h}; EXPORT=$HOME/.cache/chatcarlo/p8f_prof/p8f/export
cd /tmp && colab new -s p8fline --gpu T4 | tail -1
printf 'import os\nos.makedirs("/content/p8f",exist_ok=True)\n' | colab exec -s p8fline >/dev/null
cd $EXPORT
for f in water_bundle.npz lung_bundle.npz bone_bundle.npz lead_bundle.npz water_fluorescence.npz lung_fluorescence.npz bone_fluorescence.npz lead_fluorescence.npz bundles.json p8f_inputs.npz p8f_inputs.json p8f_source.zip p8f_manifest.json; do
  colab upload -s p8fline "$f" "/content/p8f/$f" >/dev/null 2>&1 || colab upload -s p8fline "$f" "/content/p8f/$f" >/dev/null 2>&1 || echo FAIL $f
done
for f in vm_line.py line_target.py; do colab upload -s p8fline $HERE/$f /content/$f >/dev/null 2>&1 || echo FAIL $f; done
colab exec -s p8fline -f $HERE/assemble_line.py | tail -3
