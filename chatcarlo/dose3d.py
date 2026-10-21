"""既存の線量NPZを、自己完結preview用の疎な表示データへ変換する。"""
from __future__ import annotations

import numpy as np

from . import diagnostics
from .geometry import Geometry
from .plotting import select_quantity
from .tally import VoxelGrid


class DoseOverlayError(ValueError):
    """線量入力の読込み・照合・表示データ構築の失敗。"""


def _expect(name, expected, actual):
    raise DoseOverlayError(f"{name}: 期待値 {expected}、実際の値 {actual}")


def _thin(indices, values, reference, limit):
    """桁ごとの水位配分。各組の最大を保ち、残りを一様に抽出する。"""
    groups = {}
    if len(indices):
        decades = np.floor(np.log10(values[indices] / reference)).astype(int)
        groups = {int(d): indices[decades == d] for d in np.unique(decades)}
    quotas = dict.fromkeys(groups, 0)
    pending, remaining = list(groups), min(limit, len(indices))
    while pending:
        share, extra = divmod(remaining, len(pending))
        small = [d for d in pending if len(groups[d]) <= share]
        if small:
            for d in small:
                quotas[d] = len(groups[d])
                remaining -= quotas[d]
                pending.remove(d)
        else:
            for i, d in enumerate(pending):
                quotas[d] = share + (i < extra)
            break
    rng = np.random.default_rng(20261021)
    kept, report = [], {}
    for d, ids in groups.items():
        count = quotas[d]
        if count < len(ids):
            peak = int(np.argmax(values[ids]))
            ids = np.concatenate((ids[[peak]], rng.choice(np.delete(ids, peak), count - 1, replace=False)))
        kept.extend(ids.tolist())
        report[str(d)] = {"kept": count, "original": len(groups[d])}
    return np.sort(np.asarray(kept, dtype=int)), report


def build_dose_overlay(npz_path, scene, quantity="h10") -> dict:
    """照合、分類、間引きとJSON用データ構築。輸送・NPZは変更しない。"""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return _build(data, scene, quantity)
    except DoseOverlayError:
        raise
    except Exception as exc:
        raise DoseOverlayError(f"線量.npz: 期待値 読み取り可能な正しい入力、実際の値 {exc}") from exc


def _build(data, scene, quantity):
    if quantity not in ("h10", "dose"):
        _expect("quantity", "h10 または dose", quantity)
    calibrated_key, relative_key, unit = (("h10_pSv", "h10_per_history_pSv", "pSv")
                                         if quantity == "h10" else
                                         ("dose_Gy", "dose_per_history_Gy", "Gy"))
    for key in ("origin_cm", "shape", "voxel_size_cm",
                calibrated_key if calibrated_key in data else relative_key):
        if key not in data:
            _expect(key, "キーあり", "欠落")
    size = float(data["voxel_size_cm"])
    if not np.isfinite(size) or size <= 0:
        _expect("voxel_size_cm", "正の有限値", size)
    geometry = Geometry(scene.raw["geometry"])
    grid = VoxelGrid.from_bbox(geometry.tally_bbox_min, geometry.tally_bbox_max, size)
    shape = data["shape"]
    if shape.shape != (3,) or not np.array_equal(shape, grid.shape):
        _expect("shape", grid.shape, shape.tolist())
    shape = tuple(int(x) for x in shape)
    origin = data["origin_cm"]
    if origin.shape != (3,) or not np.allclose(origin, grid.origin_cm, rtol=0, atol=1e-9):
        _expect("origin_cm", grid.origin_cm.tolist(), origin.tolist())
    values, _ = select_quantity(data, quantity)
    stat_keys = (f"rel_err_{quantity}", "n_batches", "n_batches_hit")
    present = [k for k in stat_keys if k in data]
    if present and len(present) != 3:
        _expect("統計キー", list(stat_keys), present)
    for key in (calibrated_key if calibrated_key in data else relative_key,
                *([stat_keys[0], stat_keys[2]] if present else [])):
        if data[key].shape != shape:
            _expect(f"{key} の形", shape, data[key].shape)
    if not np.all(np.isfinite(values)):
        _expect("量の値", "すべて有限", "NaN または Inf")
    values = np.asarray(values, dtype=float).ravel()
    if not np.any(values > 0):
        _expect("量の値", "正の値が1個以上", "すべて0以下")
    batches = int(data["n_batches"]) if present else None
    stats = batches is not None and batches >= 2
    R = np.asarray(data[stat_keys[0]], dtype=float).ravel() if stats else None
    hit = data["n_batches_hit"].ravel() if stats else None
    points = grid.voxel_centers()
    mats = geometry.material_at(points)
    materials, mat_codes = np.unique(mats, return_inverse=True)
    zero = values <= 0
    if stats:
        zero |= hit == 0
    src = np.asarray(scene.raw["source"]["position"])
    near = (~zero & (mats == geometry.background) &
            (np.linalg.norm(points - src, axis=1) < geometry.nearest_object_distance_cm(src)))
    why = np.zeros(len(values), dtype=np.uint8)
    if stats:
        why |= (R > diagnostics._UNRELIABLE_R_THRESHOLD).astype(np.uint8)
        why |= (hit / batches < diagnostics._UNRELIABLE_HIT_FRACTION_THRESHOLD).astype(np.uint8) * 2
        why |= (~np.isfinite(R)).astype(np.uint8) * 4
    eligible = ~zero & ~near
    masks = {"main": eligible & (why == 0), "unreliable": eligible & (why != 0)}
    reference = max(values[masks["main"]], default=0) or max(values[masks["unreliable"]], default=0)
    counts = {"zero": int(zero.sum()), "near_source_air": int(near.sum()),
              "below_range": {}, "thinned": {}, "embedded": {}}
    columns, decades = {}, {}
    for name, limit in (("main", 30000), ("unreliable", 20000)):
        mask = masks[name]
        below = mask & (values < reference * 1e-6)
        ids = np.flatnonzero(mask & ~below)
        kept, decades[name] = _thin(ids, values, reference, limit)
        counts["below_range"][name] = int(below.sum())
        counts["thinned"][name] = len(ids) - len(kept)
        counts["embedded"][name] = len(kept)
        ijk = np.unravel_index(kept, shape)
        columns[name] = {k: arr.tolist() for k, arr in zip(("ix", "iy", "iz"), ijk)}
        columns[name].update(v=[float(f"{v:.4g}") for v in values[kept]],
                             R=[round(float(r), 4) if np.isfinite(r) else None for r in R[kept]] if stats else [None] * len(kept),
                             hit=hit[kept].astype(int).tolist() if stats else [None] * len(kept),
                             why=why[kept].tolist(), mat=mat_codes[kept].tolist())
    calibrated = calibrated_key in data
    source = scene.raw["source"]
    setting = (f"CTDIvol={source['ctdi_vol_mGy']:g} mGy" if "ctdi_vol_mGy" in source else
               f"mAs={source.get('mas', '未指定')}")
    calibration_note = f".npz は校正済み。シーンの設定: {setting}" if calibrated else "相対値・校正なし"
    unit_label = f"{'H*(10)' if quantity == 'h10' else '吸収線量'} [{unit if calibrated else unit + '/history'}]"
    notes = ["外形が同じ別のシーンは見分けられない。",
             "色は turbo（低い値は青、高い値は赤）。半透明の重なりで画面の色は変わる。数値はクリックで確認。"]
    if not stats:
        notes.append("統計の情報が無いため、どのボクセルの信頼性も確かめていない")
    if quantity == "dose":
        notes.append("空気のボクセルの吸収線量は、人や検出器がいる位置の評価には使えない")
    if source.get("rotation") is not None:
        notes.append("線源近くの空気は、最初の線源の位置だけで判定している。")
    if source["field"].get("shape") == "parallel":
        notes.append("線源近くの空気は、線源面の中心で判定している。")
    unreliable_count = int(masks["unreliable"].sum())
    if unreliable_count:
        notes.append(f"灰色のボクセル（{unreliable_count} 個）は、R>0.10、寄与バッチ<25%、R が計算できない、のどれか。履歴数を増やすと減る")
    if any(counts["thinned"].values()):
        notes.append("表示は間引き済み。空間的な広がりは保つが、隣り合うボクセルが欠ける")
    if not masks["main"].any():
        if masks["unreliable"].any():
            notes.append("統計条件を満たすボクセルが無い。-n を増やすか --batch-size を下げる")
        elif np.all(near[values > 0]):
            notes.append("正の値はすべて線源のすぐ近くの空気（点線源の 1/r² による非物理的な値）にある。評価したい位置に細かいグリッドを敷く")
        else:
            notes.append(f"表示できるボクセルが無い。分類の内訳: {counts}。.npz とシーンの組み合わせを確かめる")
    main_values = columns["main"]["v"]
    # Initial colour range from the distribution of all reliable voxels (before thinning): a few
    # primary-beam voxels would otherwise push everything else into the bottom of the scale.
    # Voxels above H are drawn in the top colour; the slider still reaches the true maximum.
    full = values[masks["main"]]
    data_max = float(full.max()) if full.size else None
    vmax = float(np.percentile(full, 99)) if full.size else None
    vmin = max(float(np.percentile(full, 1)), vmax * 1e-6) if full.size else None
    if vmax is not None and vmin >= vmax:
        vmin = vmax / 10
    import matplotlib
    from matplotlib.colors import to_hex
    return dict(quantity=quantity, unit_label=unit_label, calibration_note=calibration_note,
                voxel_size_cm=size, origin_cm=origin.tolist(), shape=list(shape),
                stats_available=stats, n_batches=batches, counts=counts, decades=decades,
                **columns, vmin=vmin, vmax=vmax, data_max=data_max, materials=materials.tolist(),
                lut=[to_hex(matplotlib.colormaps["turbo"](i / 255)) for i in range(256)], notes=notes)


DOSE_HEADER = r'''<div id="doseControls" style="padding:6px 16px;display:flex;gap:12px;flex-wrap:wrap">
<label>表示下限 L <input id="doseL" type="range" step="0.1"><output id="doseLValue"></output></label>
<label>色の上限 H <input id="doseH" type="range" step="0.1"><output id="doseHValue"></output></label>
<label>不透明度 <input id="doseAlpha" type="range" min="0" max="1" step="0.01" value="0.3"></label>
<label><input id="doseGrey" type="checkbox">統計条件を満たさないボクセルも表示（灰色）</label>
</div><div id="doseBanner" style="background:#991b1b;color:white;padding:6px 16px;display:none"></div>'''

DOSE_LEGEND = r'''<div id="doseLegend" style="position:absolute;right:12px;bottom:30px;max-width:350px;max-height:55%;overflow:auto;background:var(--panel);padding:10px;font-size:12px"></div><div id="dosePick" style="position:absolute;bottom:30px;left:12px;max-width:400px;background:var(--panel);font-size:12px;white-space:pre-wrap"></div>'''

DOSE_JS = r'''
const dose = DATA.dose;
const dL = document.getElementById('doseL'), dH = document.getElementById('doseH');
const dAlpha = document.getElementById('doseAlpha'), dGrey = document.getElementById('doseGrey');
let doseRects = [], doseTime = 0;
const doseValues = dose.main.v.concat(dose.unreliable.v);
let doseMin = Infinity, doseMax = 0;
for (const v of doseValues) { doseMin = Math.min(doseMin,v); doseMax = Math.max(doseMax,v); }
const doseB = doseMax > 0 ? Math.log10(doseMax) : 0;
const doseA = doseMax > 0 ? Math.min(Math.log10(doseMin),doseB-1) : -1;
let doseLogL = dose.vmin === null ? doseA : Math.log10(dose.vmin);
let doseLogH = dose.vmax === null ? null : Math.log10(dose.vmax);
let doseAlpha = 0.3, doseGrey = false;
dL.min = doseA; dL.max = doseB-0.1;
dH.min = doseA+0.1; dH.max = doseB; dH.disabled = doseLogH === null;
const dLegend = document.getElementById('doseLegend');
function doseText(parent, text) {
  const line = document.createElement('div'); line.textContent = text; parent.appendChild(line); return line;
}
doseText(dLegend, dose.unit_label + '（' + dose.calibration_note + '）');
doseText(dLegend, dose.stats_available ? '統計条件（R≤0.10 かつ寄与バッチ≥25%）を満たす' : '統計未確認');
if (dose.data_max !== null) doseText(dLegend, '色の範囲の初期値: 統計条件を満たすボクセルの下位 1%〜上位 1% の境。H より上は最も赤い色。最大値 '+dose.data_max.toExponential(3));
const doseBar = document.createElement('div');
doseBar.style.cssText = 'display:flex;gap:8px;margin:6px 0';
const gradient = document.createElement('div');
gradient.style.cssText = 'width:18px;min-height:130px;background:linear-gradient(to top,'+dose.lut.join(',')+')';
doseBar.appendChild(gradient);
const ticks = document.createElement('div');
ticks.style.cssText='position:relative;height:150px;min-width:130px';
doseBar.appendChild(ticks); dLegend.appendChild(doseBar);
if (doseLogH === null) doseBar.style.display = 'none';
for (const [key,val] of Object.entries(dose.counts)) doseText(dLegend, key + ': ' + JSON.stringify(val));
for (const [group, rows] of Object.entries(dose.decades)) {
  for (const [decade, n] of Object.entries(rows)) doseText(dLegend, group+' '+decade+'桁: '+n.kept+'/'+n.original);
}
for (const note of dose.notes) doseText(dLegend, note);
if (!dose.stats_available) {
  const banner = document.getElementById('doseBanner');
  banner.textContent = '統計の情報が無いため、どのボクセルの信頼性も確かめていない'; banner.style.display='block';
}
function doseSync() {
  dL.value = doseLogL; dH.value = doseLogH === null ? doseB : doseLogH;
  dAlpha.value = doseAlpha; dGrey.checked = doseGrey;
  document.getElementById('doseLValue').textContent = (10**doseLogL).toExponential(3);
  document.getElementById('doseHValue').textContent = doseLogH === null ? '—' : (10**doseLogH).toExponential(3);
  ticks.textContent = '';
  if (doseLogH !== null) {
    const tick = (log,label) => {
      const line=doseText(ticks,label);
      line.style.cssText='position:absolute;transform:translateY(-50%);top:'+
        (100*(doseLogH-log)/(doseLogH-doseLogL))+'%';
    };
    tick(doseLogH,'H '+(10**doseLogH).toExponential(3));
    for (let e=Math.floor(doseLogH);e>doseLogL;e--) {
      if (e<doseLogH) tick(e,'10^'+e);
    }
    tick(doseLogL,'L '+(10**doseLogL).toExponential(3));
  }
}
function doseSet(s, redraw=true) {
  const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
  if (Number.isFinite(s.L) && s.L>0) {
    doseLogL = clamp(Math.log10(s.L),doseA,doseB-0.1);
    if (doseLogH !== null) doseLogH = Math.max(doseLogH,doseLogL+0.1);
  }
  if (doseLogH !== null && Number.isFinite(s.H) && s.H>0) {
    doseLogH = clamp(Math.log10(s.H),doseA+0.1,doseB);
    doseLogL = Math.min(doseLogL,doseLogH-0.1);
  }
  doseLogL = clamp(doseLogL,doseA,doseB-0.1);
  if (doseLogH !== null) doseLogH = clamp(Math.max(doseLogH,doseLogL+0.1),doseA+0.1,doseB);
  if (Number.isFinite(s.opacity)) doseAlpha = clamp(s.opacity,0,1);
  if (typeof s.grey === 'boolean') doseGrey = s.grey;
  doseSync(); if (redraw) draw();
}
dL.oninput = () => doseSet({L:10**Number(dL.value)});
dH.oninput = () => doseSet({H:10**Number(dH.value)});
dAlpha.oninput = () => doseSet({opacity:Number(dAlpha.value)});
dGrey.onchange = () => doseSet({grey:dGrey.checked});
function drawDose() {
  const start = performance.now();
  const lo = dose.origin_cm, size = dose.voxel_size_cm;
  const hi = lo.map((v,k)=>v+dose.shape[k]*size);
  const corners = [];
  for (let z=0;z<2;z++) for(let y=0;y<2;y++) for(let x=0;x<2;x++) corners.push([x?hi[0]:lo[0],y?hi[1]:lo[1],z?hi[2]:lo[2]]);
  for (let i=0;i<8;i++) for (const bit of [1,2,4]) if (!(i&bit)) seg(corners[i],corners[i|bit],'#94a3b8',1,[4,4]);
  const rects = [];
  if (doseAlpha>0) for (const group of (doseGrey ? ['main','unreliable'] : ['main'])) {
    const c = dose[group];
    for (let i=0;i<c.v.length;i++) {
      if (c.v[i] < 10**doseLogL) continue;
      const center = [lo[0]+(c.ix[i]+0.5)*size,lo[1]+(c.iy[i]+0.5)*size,lo[2]+(c.iz[i]+0.5)*size];
      const p = project(center); if (!p) continue;
      const edge = size * 1.2 * Math.min(cv.clientWidth,cv.clientHeight)/(2*Math.tan(0.35)*p[2]);
      const x=p[0]-edge/2,y=p[1]-edge/2;
      if (x+edge<0 || y+edge<0 || x>cv.clientWidth || y>cv.clientHeight) continue;
      const index = doseLogH === null ? 0 : Math.round(255*Math.max(0,Math.min(1,(Math.log10(c.v[i])-doseLogL)/(doseLogH-doseLogL))));
      rects.push({x,y,edge,depth:p[2],group,i,center,color:group==='unreliable'?'#9ca3af':dose.lut[index]});
    }
  }
  rects.sort((a,b)=>b.depth-a.depth);
  ctx.globalAlpha = doseAlpha;
  for (const r of rects) { ctx.fillStyle=r.color; ctx.fillRect(r.x,r.y,r.edge,r.edge); }
  ctx.globalAlpha=1; doseRects=rects; doseTime=performance.now()-start;
}
function dosePick(clientX,clientY) {
  const bounds = cv.getBoundingClientRect(), x=clientX-bounds.left,y=clientY-bounds.top;
  if (x<0 || y<0 || x>bounds.width || y>bounds.height) return null;
  for (let j=doseRects.length-1;j>=0;j--) {
    const r=doseRects[j];
    if (x<r.x || x>r.x+r.edge || y<r.y || y>r.y+r.edge) continue;
    const c=dose[r.group], i=r.i;
    return {ix:c.ix[i],iy:c.iy[i],iz:c.iz[i],v:c.v[i],R:c.R[i],hit:c.hit[i],why:c.why[i],
            material:dose.materials[c.mat[i]],center_cm:r.center,unit_label:dose.unit_label,group:r.group};
  }
  return null;
}
let doseDown=null, doseMoved=false;
cv.addEventListener('mousedown',e=>{doseDown=[e.clientX,e.clientY];doseMoved=false;});
addEventListener('mousemove',e=>{if(doseDown && Math.hypot(e.clientX-doseDown[0],e.clientY-doseDown[1])>=4)doseMoved=true;});
addEventListener('mouseup',e=>{
  if (doseDown && !doseMoved && Math.hypot(e.clientX-doseDown[0],e.clientY-doseDown[1])<4 && e.target===cv) {
    const p=dosePick(e.clientX,e.clientY), box=document.getElementById('dosePick');
    box.textContent='';
    if(p) {
      let converted='';
      if (!dose.unit_label.includes('/history')) {
        const base=p.v*(dose.quantity==='h10'?1e-12:1), unit=dose.quantity==='h10'?'Sv':'Gy';
        const [factor,prefix]=base>=1?[1,'']:base>=1e-3?[1e3,'m']:base>=1e-6?[1e6,'µ']:[1e9,'n'];
        converted='（= '+(base*factor).toPrecision(4)+' '+prefix+unit+'）';
      }
      const reasons=[];
      if(p.why&1)reasons.push('R が大きい');if(p.why&2)reasons.push('寄与バッチが少ない');if(p.why&4)reasons.push('R が計算できない');
      box.textContent=p.v+' '+dose.unit_label+converted+'\nR='+p.R+'、寄与バッチ='+p.hit+'/'+dose.n_batches+
        '\n統計不足の理由: '+(reasons.join('、')|| (dose.stats_available?'該当なし':'未確認'))+
        '\n材料: '+p.material+'、中心 [cm]: '+p.center_cm.join(', ')+'\n添字: '+[p.ix,p.iy,p.iz].join(', ');
    }
  }
  doseDown=null;
});
window.__dose3d={pick:dosePick,state:()=>({L:10**doseLogL,H:doseLogH===null?null:10**doseLogH,opacity:doseAlpha,grey:doseGrey,drawn:doseRects.length,draw_ms:doseTime}),setState:s=>{doseSet(s);return window.__dose3d.state();}};
doseSet({},false);
'''
