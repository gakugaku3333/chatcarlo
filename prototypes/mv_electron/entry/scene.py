"""Strict, deliberately small therapy scene allowlist (no implicit defaults)."""
from pathlib import Path
import yaml


class SceneError(ValueError):
    pass


def validate(raw):
    if isinstance(raw, dict) and raw.get("scoring") == "seed_ratio":
        from .seed_scene import validate as validate_seed
        return validate_seed(raw)

    def fail(path, detail):
        raise SceneError(f'{path}: {detail}。この入口で計算できる条件: mohan10 または mohan4、水30 cm、10×10 cm²、SSD 100 cm、pdd_ocr')

    def keys(value, expected, path):
        if not isinstance(value, dict):
            fail(path, 'mapping が必要')
        missing = set(expected)-set(value)
        unknown = set(value)-set(expected)
        if missing:
            fail('.'.join(filter(None, (path, sorted(missing)[0]))), '必須キーがありません')
        if unknown:
            fail('.'.join(filter(None, (path, str(next(iter(unknown)))))), '未知のキー')

    keys(raw, ('domain', 'source', 'phantom', 'scoring', 'run'), '')
    keys(raw['source'], ('spectrum', 'field_cm', 'ssd_cm'), 'source')
    keys(raw['phantom'], ('material', 'thickness_cm'), 'phantom')
    keys(raw['run'], ('histories', 'seed', 'engine'), 'run')
    expected = {'domain': 'therapy_mv', 'scoring': 'pdd_ocr', 'source.spectrum': ('mohan10', 'mohan4'),
                'source.field_cm': [10, 10], 'source.ssd_cm': 100,
                'phantom.material': 'water', 'phantom.thickness_cm': 30}
    for path, wanted in expected.items():
        value = raw
        for key in path.split('.'):
            value = value[key]
        numeric = isinstance(wanted, (int, list))
        vals = value if isinstance(value, list) else [value]
        choices = wanted if isinstance(wanted, tuple) else None
        if (value not in choices if choices else value != wanted) or (numeric and any(type(v) not in (int, float) for v in vals)):
            fail(path, ' | '.join(choices)+' のみ' if choices else f'{wanted} のみ')
    run = raw['run']
    if run['engine'] not in ('colab_t4', 'cpu_local'):
        fail('run.engine', 'colab_t4 | cpu_local のみ')
    cap = 240000000 if run['engine'] == 'colab_t4' else 200000
    n, seed = run['histories'], run['seed']
    if type(n) is not int or not 0 < n <= cap or n % 100:
        fail('run.histories', f'100 の倍数の整数、100〜{cap} が必要')
    if type(seed) is not int or not 0 <= seed < 2**64:
        fail('run.seed', '0 以上 2⁶⁴ 未満の整数が必要')
    return raw


def load(path):
    return validate(yaml.safe_load(Path(path).read_text()))
