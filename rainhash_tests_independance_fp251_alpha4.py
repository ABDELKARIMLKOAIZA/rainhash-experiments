from rainhash_sponge_fp251_alpha4_8tours import RAINHASH_MODULE, RAINHASH_SOURCE

TEST_SOURCE = r'''
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
from itertools import combinations
from time import perf_counter
import numpy as np
import scipy
from scipy.stats import chi2
P, D, K = (251, 32, 25)
PARTITIONS = [(1, 0, 17, 5), (17, 5, 1, 0), (53, 71, 127, 200), (29, 101, 89, 37)]
SEED_DETECTION = 2026092801
SEED_VALIDATION = 2026092802
SEED_PROJECTIONS = 2026092803
SEED_PERMUTATIONS = 2026092804
SEED_CONTROLE = 2026092805
ALPHA_PHASE = 0.025

def sha(data):
    return hashlib.sha256(data).hexdigest()

def json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')

def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def save_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, tuple, dict)) else v for k, v in row.items()})
    tmp.replace(path)

def holm(rows, alpha):
    if not rows:
        return
    ps = np.asarray([r['p_value'] for r in rows], dtype=float)
    if np.any(~np.isfinite(ps)) or np.any((ps < 0) | (ps > 1)):
        raise ValueError('p-value invalide')
    order = np.argsort(ps, kind='mergesort')
    adjusted = np.minimum(1.0, np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)))
    for idx, value in zip(order, adjusted):
        rows[int(idx)]['p_holm'] = float(value)
        rows[int(idx)]['rejet_holm'] = bool(value <= alpha)

def make_specs():
    specs = []

    def add(famille, **kw):
        specs.append(dict(id=len(specs), famille=famille, **kw))
    for i in range(D):
        add('marginale', indices=[i], coefficients=[1])
    for i, j in combinations(range(D), 2):
        for part in range(len(PARTITIONS)):
            add('paire_groupee', indices=[i, j], partition=part)
        add('difference', indices=[i, j], coefficients=[1, -1])
        add('somme', indices=[i, j], coefficients=[1, 1])
    seen = set()

    def projection(indices, coeffs):
        pairs = sorted(zip(map(int, indices), map(int, coeffs)))
        inv = pow(pairs[0][1] % P, P - 2, P)
        indices = [i for i, _ in pairs]
        coeffs = [c * inv % P for _, c in pairs]
        key = (tuple(indices), tuple(coeffs))
        if key in seen:
            return False
        seen.add(key)
        add('projection_' + str(len(indices)), indices=indices, coefficients=coeffs)
        return True
    motifs = [(1, 1, -1), (1, 2, -3), (1, -1, 2), (2, 1, -1)]
    for block in range(4):
        for i in range(8):
            projection([8 * block + (i + j) % 8 for j in range(3)], motifs[i % 4])
    for i in range(8):
        for coeffs in ((1, -1, 1, -1), (1, 2, -1, -2)):
            projection([i + 8 * j for j in range(4)], coeffs)
    rng = np.random.default_rng(SEED_PROJECTIONS)
    for weight in (3, 4):
        count = 0
        while count < 32:
            count += int(projection(rng.choice(D, weight, replace=False), rng.integers(1, P, weight)))
    return specs

def uniform_test(values):
    counts = np.bincount(values, minlength=P).astype(float)
    expected = len(values) / float(P)
    stat = float(np.sum((counts - expected) ** 2 / expected))
    return dict(statistique=stat, ddl=P - 1, p_value=float(chi2.sf(stat, P - 1)), attendu_min=expected, variation_totale=float(0.5 * np.sum(np.abs(counts / len(values) - 1 / P))))

def grouped_test(x, y, part, i, j):
    ax, bx, ay, by = PARTITIONS[part]
    gx = K * ((ax * x + bx + 37 * i + 13 * i * i) % P) // P
    gy = K * ((ay * y + by + 61 * j + 19 * j * j) % P) // P
    table = np.bincount(gx * K + gy, minlength=K * K).reshape(K, K).astype(float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    nr, nc = table.shape
    if min(nr, nc) < 2:
        return dict(statistique=0.0, ddl=0, p_value=1.0, attendu_min=0.0, cramer_v=0.0, information_mutuelle=0.0, degenere=True)
    n = table.sum()
    exp = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    stat = float(np.sum((table - exp) ** 2 / exp))
    mask = table > 0
    mi = float(np.sum(table[mask] / n * np.log(table[mask] / exp[mask])))
    return dict(statistique=stat, ddl=(nr - 1) * (nc - 1), p_value=float(chi2.sf(stat, (nr - 1) * (nc - 1))), attendu_min=float(exp.min()), cramer_v=float(np.sqrt(stat / (n * min(nr - 1, nc - 1)))), information_mutuelle=mi, degenere=False)

def evaluate(outputs, spec, length, strict=True):
    if spec['famille'] == 'paire_groupee':
        i, j = spec['indices']
        result = grouped_test(outputs[:, i].astype(np.int64), outputs[:, j].astype(np.int64), spec['partition'], i, j)
    else:
        values = np.zeros(len(outputs), dtype=np.int64)
        for i, c in zip(spec['indices'], spec['coefficients']):
            values += int(c) * outputs[:, i].astype(np.int64)
        result = uniform_test(values % P)
    if strict and result['attendu_min'] < 5:
        raise ValueError('Effectif attendu < 5 : ne pas interpréter le khi-deux. Longueur {}, test {} : {}'.format(length, spec['id'], result))
    return dict(longueur=int(length), **spec, **result)

def permutation_mi(x, y, B, rng):
    x, y = (np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64))
    n = len(x)

    def entropy(counts):
        v = counts[counts > 0] / float(n)
        return float(-np.sum(v * np.log(v)))
    marginal = entropy(np.bincount(x, minlength=P)) + entropy(np.bincount(y, minlength=P))

    def statistic(z):
        return marginal - entropy(np.bincount(x * P + z, minlength=P * P))
    observed = statistic(y)
    exceed, total = (0, 0.0)
    tol = 100 * np.finfo(float).eps * max(1.0, abs(observed))
    for _ in range(B):
        value = statistic(rng.permutation(y))
        exceed += int(value >= observed - tol)
        total += value
    return dict(information_mutuelle=observed, moyenne_mi_permutations=total / B, mi_moins_moyenne_permutations=observed - total / B, permutations=int(B), depassements=int(exceed), p_value=(exceed + 1.0) / (B + 1.0))

def select_validation(rows, specs, top):
    chosen = {(r['longueur'], r['id']) for r in rows if r['rejet_holm']}
    for family in sorted({r['famille'] for r in rows}):
        candidates = sorted((r for r in rows if r['famille'] == family), key=lambda r: (r['p_value'], r['longueur'], r['id']))
        chosen.update(((r['longueur'], r['id']) for r in candidates[:top]))
    selected = [dict(longueur=l, **specs[idx]) for l, idx in sorted(chosen)]
    pairs = sorted({(s['longueur'], *s['indices']) for s in selected if s['famille'] == 'paire_groupee'})
    return (selected, pairs)

def load_hash(path):
    module = RAINHASH_MODULE
    if (int(module.p), int(module.ALPHA_L), int(module.NB_TOURS), int(module.TAUX_SPONGE), int(module.CAPACITE_SPONGE), int(module.TAILLE_EMPREINTE_SPONGE)) != (251, 4, 8, 4, 4, 4):
        raise ValueError('Le hachage doit utiliser p=251, alpha=4, huit tours, r=c=sortie=4')
    witnesses = []
    for l in (0, 1, 4, 5):
        msg = [[8 * j + i + 1 for i in range(8)] for j in range(l)]
        witnesses.append([int(v) for v in module.coordonnees_empreinte_F251(module.hachage_sponge_pi_T3(msg))])
    return (module, witnesses)

def obtain_outputs(module, length, phase, n, root, hash_id):
    seed = SEED_DETECTION if phase == 'detection' else SEED_VALIDATION
    meta = dict(hash_id=hash_id, longueur=length, phase=phase, n=n, seed=seed, generateur='numpy.PCG64/SeedSequence([seed,longueur])', version_numpy=np.__version__)
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    path = cache / '{}_L{:02d}.npy'.format(phase, length)
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        stored = json.loads(metadata.read_text())
        if stored['parametres'] != meta or stored['sha256_npy'] != sha(path.read_bytes()):
            raise ValueError('Cache incompatible ou endommagé : ' + str(path))
        data = np.load(path, allow_pickle=False)
        if data.shape != (n, D) or data.dtype != np.uint8 or np.any(data >= P):
            raise ValueError('Cache invalide : ' + str(path))
        print('Cache validé :', path.name, flush=True)
        return data
    rng = np.random.default_rng(np.random.SeedSequence([seed, length]))
    out = np.empty((n, D), dtype=np.uint8)
    start = perf_counter()
    for k in range(n):
        msg = rng.integers(0, P, size=(length, 8), dtype=np.int64).tolist()
        h = module.coordonnees_empreinte_F251(module.hachage_sponge_pi_T3(msg))
        values = [int(v) for v in h]
        if len(values) != D or any((v < 0 or v >= P for v in values)):
            raise ValueError('Condensat invalide')
        out[k] = values
        if k + 1 == min(100, n) or (k + 1) % max(1, n // 10) == 0:
            elapsed = perf_counter() - start
            print('{} L={} : {}/{}, reste estimé {:.1f} min'.format(phase, length, k + 1, n, elapsed * (n - k - 1) / (k + 1) / 60), flush=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, out, allow_pickle=False)
    tmp.replace(path)
    save_json(metadata, dict(parametres=meta, sha256_npy=sha(path.read_bytes())))
    return out

def targeted_power(n, m, repetitions):
    rng = np.random.default_rng(SEED_CONTROLE)
    rows = []
    threshold = ALPHA_PHASE / m
    for scenario in ('nul', 'paire_0.5pct', 'triple_0.5pct', 'quadruple_0.5pct'):
        rejected = 0
        for _ in range(repetitions):
            x = rng.integers(0, P, size=(n, 4), dtype=np.int64)
            mask = rng.random(n) < 0.005
            if scenario == 'paire_0.5pct':
                x[mask, 1] = x[mask, 0]
            elif scenario == 'triple_0.5pct':
                x[mask, 2] = (x[mask, 0] + x[mask, 1]) % P
            elif scenario == 'quadruple_0.5pct':
                x[mask, 3] = (x[mask, 0] + x[mask, 1] + x[mask, 2]) % P
            if scenario in ('nul', 'paire_0.5pct'):
                z = x[:, 0] - x[:, 1]
            elif scenario == 'triple_0.5pct':
                z = x[:, 0] + x[:, 1] - x[:, 2]
            else:
                z = x[:, 0] + x[:, 1] + x[:, 2] - x[:, 3]
            rejected += int(uniform_test(z % P)['p_value'] <= threshold)
        rate = rejected / repetitions
        z95 = 1.959963984540054
        den = 1 + z95 * z95 / repetitions
        center = (rate + z95 * z95 / (2 * repetitions)) / den
        half = z95 * math.sqrt(rate * (1 - rate) / repetitions + z95 * z95 / (4 * repetitions ** 2)) / den
        rows.append(dict(scenario=scenario, repetitions=repetitions, rejets=rejected, taux=rate, ic95_wilson_inf=max(0.0, center - half), ic95_wilson_sup=min(1.0, center + half), seuil=threshold, portee='contraste_cible_au_seuil_Bonferroni_pas_pipeline_complet'))
    return rows

def run(args):
    mode = args.mode
    article = mode == 'article'
    n, lengths, top = (50200, list(range(1, 17)), 5) if article else (1004, [1, 4], 1)
    hash_path = None
    module, witnesses = load_hash(hash_path)
    specs = make_specs()
    if not article:
        specs = [s for family in sorted({s['famille'] for s in specs}) for s in [s for s in specs if s['famille'] == family][:2]]
        specs = [dict(s, id=i) for i, s in enumerate(specs)]
    try:
        from sage.env import SAGE_VERSION
    except ImportError:
        SAGE_VERSION = 'inconnu'
    hash_id = sha(json_bytes(dict(source_sha256=sha(RAINHASH_SOURCE.encode('utf-8')), temoins=witnesses)))
    protocol = dict(version='independance_alpha4_v1', mode=mode, n=n, longueurs=lengths, hash_id=hash_id, hachage_sha256=sha(RAINHASH_SOURCE.encode('utf-8')), temoins=witnesses, script_sha256=sha(TEST_SOURCE.encode('utf-8')), alpha_detection=ALPHA_PHASE, alpha_validation=ALPHA_PHASE, top_par_famille=top, partitions=PARTITIONS, specifications=specs, graines=dict(detection=SEED_DETECTION, validation=SEED_VALIDATION, projections=SEED_PROJECTIONS, permutations=SEED_PERMUTATIONS, controle=SEED_CONTROLE), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sage=SAGE_VERSION, repetitions_controle=200 if article else 3, convention_indices='indices de coordonnées 0,...,31', p_values='khi-deux asymptotique ; permutation Monte-Carlo à B fixe avec +1')
    root = Path(args.sortie) / '{}_{}'.format(mode, sha(json_bytes(protocol))[:16])
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'protocole.json', protocol)
    m = len(lengths) * len(specs)
    print('Résultats :', root.resolve(), flush=True)
    print('Tests de détection :', m, '; mode :', mode, flush=True)
    if not article:
        print('Mode vérification.', flush=True)
    rows = []
    for length in lengths:
        outputs = obtain_outputs(module, length, 'detection', n, root, hash_id)
        rows.extend((evaluate(outputs, s, length, strict=article) for s in specs))
        save_csv(root / 'detection_partielle.csv', rows)
    holm(rows, ALPHA_PHASE)
    save_csv(root / 'detection.csv', rows)
    selected, pairs = select_validation(rows, specs, top)
    mv = len(selected) + len(pairs)
    B = max(9999, math.ceil(mv / ALPHA_PHASE) - 1) if article else 19
    save_json(root / 'selection_validation.json', dict(tests=selected, paires_non_groupees=pairs, total_tests=mv, B_fixe=B, seuil_holm_initial=ALPHA_PHASE / mv, resolution_p_monte_carlo=1 / (B + 1)))
    validation = []
    for length in sorted({s['longueur'] for s in selected}):
        outputs = obtain_outputs(module, length, 'validation', n, root, hash_id)
        for s in selected:
            if s['longueur'] == length:
                validation.append(evaluate(outputs, specs[s['id']], length, strict=article))
        for l, i, j in pairs:
            if l == length:
                print('Permutation : L={}, paire ({},{}), B={}'.format(l, i, j, B), flush=True)
                rng = np.random.default_rng(np.random.SeedSequence([SEED_PERMUTATIONS, l, i, j]))
                result = permutation_mi(outputs[:, i], outputs[:, j], B, rng)
                validation.append(dict(longueur=l, id='MC_{}_{}'.format(i, j), famille='paire_non_groupee', indices=[i, j], **result))
        save_csv(root / 'validation_partielle.csv', validation)
    holm(validation, ALPHA_PHASE)
    save_csv(root / 'validation.csv', validation)
    power = targeted_power(n, m, 200 if article else 3)
    save_csv(root / 'controle_synthetique.csv', power)
    summary = dict(mode=mode, utilisable_article=article, sorties_detection=n * len(lengths), sorties_validation=n * len({s['longueur'] for s in selected}), tests_detection=len(rows), tests_validation=len(validation), B_permutations=B, rejets_detection=sum((r['rejet_holm'] for r in rows)), rejets_validation=sum((r['rejet_holm'] for r in validation)), familles={})
    for family in sorted({r['famille'] for r in rows + validation}):
        a = [r for r in rows if r['famille'] == family]
        b = [r for r in validation if r['famille'] == family]
        summary['familles'][family] = dict(tests_detection=len(a), tests_validation=len(b), rejets_detection=sum((r['rejet_holm'] for r in a)), rejets_validation=sum((r['rejet_holm'] for r in b)))
    save_json(root / 'resume.json', summary)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if not article:
        text += '\nMode vérification.\n'
    elif summary['rejets_validation']:
        text += '\nAnomalie détectée en validation.\n'
    elif summary['rejets_detection']:
        text += '\nAnomalies de détection non reproduites en validation.\n'
    else:
        text += '\nAucune anomalie détectée par les tests effectués aux seuils indiqués.\n'
    (root / 'resume.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


import builtins as B
import math as _math

def holm_python(rows, alpha):
    print("Holm : début sur {} tests.".format(B.len(rows)), flush=True)
    if not rows:
        return
    probabilites = [B.float(r["p_value"]) for r in rows]
    for p in probabilites:
        if not _math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p-value invalide : {}".format(p))
    ordre = B.sorted(B.range(B.len(rows)), key=lambda i: probabilites[i])
    maximum = 0.0
    nombre = B.len(rows)
    for rang, indice in B.enumerate(ordre):
        ajustee = B.min(1.0, (nombre - rang) * probabilites[indice])
        maximum = B.max(maximum, ajustee)
        rows[indice]["p_holm"] = maximum
        rows[indice]["rejet_holm"] = B.bool(maximum <= alpha)
    print("Holm : terminé.", flush=True)

holm = holm_python

if "_evaluate_original" not in globals():
    _evaluate_original = evaluate
    _obtain_original = obtain_outputs
    _select_original = select_validation
    _power_original = targeted_power

def evaluate(outputs, spec, length, strict=True):
    if spec["id"] == 0:
        print("Évaluation : début de L={}.".format(length), flush=True)
    resultat = _evaluate_original(outputs, spec, length, strict=strict)
    if spec["id"] == 3119:
        print("Évaluation : 3120 tests terminés pour L={}.".format(length), flush=True)
    return resultat

def obtain_outputs(module, length, phase, n, root, hash_id):
    print("Sorties : phase={}, L={}.".format(phase, length), flush=True)
    return _obtain_original(module, length, phase, n, root, hash_id)

def select_validation(rows, specs, top):
    print("Sélection de la validation : début.", flush=True)
    resultat = _select_original(rows, specs, top)
    print("Sélection terminée : {} tests et {} paires Monte-Carlo.".format(B.len(resultat[0]), B.len(resultat[1])), flush=True)
    return resultat

def targeted_power(n, m, repetitions):
    print("Contrôle synthétique : début.", flush=True)
    resultat = _power_original(n, m, repetitions)
    print("Contrôle synthétique : terminé.", flush=True)
    return resultat

print("Correction installée.")
'''

TEST_NAMESPACE = {
    '__name__': 'test_alpha4_notebook',
    'RAINHASH_MODULE': RAINHASH_MODULE,
    'RAINHASH_SOURCE': RAINHASH_SOURCE,
    'TEST_SOURCE': TEST_SOURCE,
}
exec(compile(TEST_SOURCE, '<test_independance_alpha4>', 'exec'), TEST_NAMESPACE)
print('Protocole chargé.')

from argparse import Namespace

if __name__ == '__main__':
    TEST_NAMESPACE['run'](Namespace(
        mode='article',
        sortie='resultats_independance_alpha4'
    ))
