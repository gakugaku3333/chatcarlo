"""P9c entry allowlist; no implicit defaults or coercion."""
from .scene import SceneError


def validate(raw):
    def fail(path, message):
        raise SceneError(f'{path}: {message}（seed_ratio: P9c 固定条件のみ）')

    def keys(value, names, path):
        if not isinstance(value, dict):
            fail(path, 'mapping が必要')
        for key in names:
            if key not in value:
                fail(f'{path}.{key}', '必須キーがありません')
        for key in value:
            if key not in names:
                fail(f'{path}.{key}', '未知のキー')

    keys(raw, ('domain', 'scoring', 'source', 'body', 'target', 'seeds', 'run'), 'scene')
    fixed = dict(domain='therapy_mv', scoring='seed_ratio',
                 source=dict(spectrum='mohan10', sad_cm=100, gantry_deg=[0,45,90,135,225,270,315], field='prostate_projection'),
                 body=dict(material='water', ellipse_cm=[36,24], length_cm=30),
                 target=dict(prostate_cm=[4.5,3.0,3.5], rectum_diameter_cm=3),
                 seeds=dict(preset='p9a_58', material='ag_ti_mixture'))

    def equal(value, wanted, path):
        if isinstance(wanted, dict):
            keys(value, wanted, path)
            for k, v in wanted.items():
                equal(value[k], v, f'{path}.{k}')
        elif isinstance(wanted, list):
            if not isinstance(value, list) or len(value) != len(wanted):
                fail(path, f'{wanted} のみ')
            for i, v in enumerate(wanted):
                equal(value[i], v, f'{path}[{i}]')
        elif isinstance(wanted, (float, int)):
            if type(value) not in (float, int) or value != wanted:
                fail(path, f'{wanted} のみ（数値が必要）')
        elif type(value) is not str or value != wanted:
            fail(path, f'{wanted} のみ')
    for k, v in fixed.items():
        equal(raw[k], v, k)
    keys(raw['run'], ('histories_per_field', 'seed', 'engine'), 'run')
    run = raw['run']
    equal(run['engine'], 'colab_t4', 'run.engine')
    n = run['histories_per_field']
    if type(n) is not int or not 100 <= n <= 7142900 or n % 100:
        fail('run.histories_per_field', '100〜7142900 の100の倍数の整数が必要')
    if type(run['seed']) is not int or not 0 <= run['seed'] < 2**58:
        fail('run.seed', '0 以上 2⁶¹/8 未満の整数が必要')
    return raw


def prostate_diameters(scene):
    """Scene [left-right, anterior-posterior, head-foot] -> kernel [x,y,z]."""
    x, z, y = scene['target']['prostate_cm']
    return [x, y, z]
