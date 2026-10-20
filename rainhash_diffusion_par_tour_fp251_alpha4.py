from rainhash_sponge_fp251_alpha4_8tours import RAINHASH_MODULE, RAINHASH_SOURCE

                                                              

_SOURCE_TOURS_JUPYTER = r'''
import csv, gc, hashlib, json, math, os, time
from pathlib import Path
from time import perf_counter
import numpy as np
from IPython.display import display
P = 251
TOURS = 8
DELTAS = [1, 2, 127]
FAMILLES = [('uniforme',256), ('Fp',64), ('T1',64), ('T2',64),
            ('colonne_unique',64), ('colonnes_repetees',64)]
GRAINE = 2026092931
ALPHA_GLOBAL = 0.05
T0_TABLE = None
CAMPAGNE = None
DOSSIER = None
EN_COURS = False
def canonique(objet):
    return json.dumps(objet, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def empreinte(texte):
    return hashlib.sha256(texte.encode('utf-8')).hexdigest()

def graine(*elements):
    return int(empreinte(canonique(elements))[:32], 16)

def duree(secondes):
    if secondes is None:
        return 'estimation en cours'
    secondes = max(0, int(secondes))
    h, reste = divmod(secondes, 3600)
    m, s = divmod(reste, 60)
    return '{} h {:02d} min {:02d} s'.format(h, m, s)

def ecrire_json(chemin, objet):
    chemin = Path(chemin)
    temporaire = chemin.with_name(chemin.name + '.tmp')
    temporaire.write_text(json.dumps(objet, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temporaire), str(chemin))

def ecrire_csv(chemin, lignes):
    if not lignes:
        return
    chemin = Path(chemin)
    temporaire = chemin.with_name(chemin.name + '.tmp')
    champs = sorted(set().union(*(r.keys() for r in lignes)))
    with temporaire.open('w', newline='', encoding='utf-8') as fichier:
        writer = csv.DictWriter(fichier, fieldnames=champs)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({k: canonique(ligne[k]) if isinstance(ligne.get(k), (dict, list, tuple)) else ligne.get(k, '') for k in champs})
    os.replace(str(temporaire), str(chemin))

def correction_holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.asarray([], dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError('p-values invalides')
    ordre = np.argsort(p_values, kind='mergesort')
    ajustees = np.empty(m, dtype=float)
    maximum = 0.0
    for rang, indice in enumerate(ordre):
        valeur = min(1.0, (m - rang) * float(p_values[indice]))
        maximum = max(maximum, valeur)
        ajustees[indice] = maximum
    return ajustees

def initialiser_table_T0():
    global T0_TABLE
    if T0_TABLE is None:
        module = RAINHASH_MODULE
        print('Construction de la table des 251 éléments de F_251...', flush=True)
        T0_TABLE = tuple((module.T0(i) for i in range(P)))

def convertir(etat):
    if len(etat) != 8 or any((len(colonne) != 8 for colonne in etat)):
        raise ValueError('Un état doit contenir 8 éléments de K, soit 64 coordonnées.')
    valeurs = [[int(x) for x in colonne] for colonne in etat]
    if any((x < 0 or x >= P for ligne in valeurs for x in ligne)):
        raise ValueError('Coordonnée hors de F_251.')
    return np.asarray(valeurs, dtype=np.int64)

def etat_sage(x):
    initialiser_table_T0()
    return [[T0_TABLE[int(v)] for v in ligne] for ligne in x]

def construire_paire_sage(x, position, delta):
    base = etat_sage(x)
    perturbe = [ligne[:] for ligne in base]
    i, j = divmod(int(position), 8)
    valeur = (int(x[i, j]) + int(delta)) % P
    perturbe[i][j] = T0_TABLE[valeur]
    return (base, perturbe)

def generer_etat(rng, famille):
    if famille == 'uniforme':
        return rng.integers(0, P, size=(8, 8), dtype=np.int64)
    x = np.zeros((8, 8), dtype=np.int64)
    if famille in ('Fp', 'T1', 'T2'):
        dimension = {'Fp': 1, 'T1': 2, 'T2': 4}[famille]
        x[:, :dimension] = rng.integers(0, P, size=(8, dimension), dtype=np.int64)
    elif famille == 'colonne_unique':
        indice = int(rng.integers(0, 8))
        x[indice] = rng.integers(0, P, size=8, dtype=np.int64)
    elif famille == 'colonnes_repetees':
        ligne = rng.integers(0, P, size=8, dtype=np.int64)
        x[:] = ligne
    else:
        raise ValueError('Famille inconnue : ' + famille)
    return x

def trace_paire(x, position, delta, suivi=None, compteur=0):

    module = RAINHASH_MODULE
    x_sage, y_sage = construire_paire_sage(x, position, delta)
    a = module.L_sans_MDS(x_sage)
    b = module.L_sans_MDS(y_sage)
    del x_sage
    del y_sage
    a_np = convertir(a)
    b_np = convertir(b)
    differences = np.empty((TOURS, 64), dtype=np.uint8)
    actives = np.empty(TOURS, dtype=np.uint8)
    for r in range(TOURS):
        actives[r] = np.count_nonzero(np.any(a_np != b_np, axis=1))
        a = module.tour_direct_T3(a, module.CONSTANTES_TOURS[r])
        b = module.tour_direct_T3(b, module.CONSTANTES_TOURS[r])
        a_np = convertir(a)
        b_np = convertir(b)
        diff = (b_np - a_np) % P
        differences[r] = diff.reshape(64).astype(np.uint8, copy=False)
        if not np.any(differences[r]):
            raise ArithmeticError('Deux états distincts ont fusionné au tour {}.'.format(r + 1))
        if suivi is not None:
            compteur += 1
            suivi.avancer(compteur)
    del a
    del b
    del a_np
    del b_np
    return (differences, actives, compteur)

def verifier():
    module = RAINHASH_MODULE
    if int(module.T0.cardinality()) != 251 or int(module.ALPHA_L) != 4 or int(module.NB_TOURS) != 8:
        raise ValueError('Le module doit utiliser F_251, alpha=4 et huit tours.')
    if len(module.CONSTANTES_TOURS) != 8:
        raise ValueError('Huit constantes de tours sont requises.')
    initialiser_table_T0()
    temoins = []
    for numero in range(3):
        x = ((np.arange(64).reshape(8, 8) * 17 + numero * 31) % P).astype(np.int64)
        direct = module.permutation_complete_T3(etat_sage(x))
        z = module.L_sans_MDS(etat_sage(x))
        for c in module.CONSTANTES_TOURS:
            z = module.tour_direct_T3(z, c)
        if not np.array_equal(convertir(z), convertir(direct)):
            raise AssertionError('La composition des huit tours diffère de la permutation fournie.')
        inverse = module.permutation_inverse_T3(direct)
        if not np.array_equal(convertir(inverse), x):
            raise AssertionError('Échec de la vérification de l’inverse.')
        temoins.append(convertir(direct).tolist())
        del direct
        del inverse
        del z
        gc.collect()
    return temoins

def checksum_statistiques(changements, histogramme, somme_actives, min_actives, max_actives, min_cumul):
    h = hashlib.sha256()
    tableaux = (changements, histogramme, somme_actives, min_actives, max_actives, min_cumul)
    for tableau in tableaux:
        tableau = np.ascontiguousarray(tableau)
        h.update(str(tableau.dtype).encode('ascii'))
        h.update(canonique(list(tableau.shape)).encode('ascii'))
        h.update(tableau.tobytes())
    return h.hexdigest()

def charger_cache_configuration(chemin, cle, n):
    with np.load(str(chemin), allow_pickle=False) as archive:
        campagne = str(archive['campagne'].item())
        cle_stockee = str(archive['cle'].item())
        n_stocke = int(archive['n'].item())
        checksum = str(archive['sha256'].item())
        changements = archive['changements'].copy()
        histogramme = archive['histogramme'].copy()
        somme_actives = archive['somme_actives'].copy()
        min_actives = archive['min_actives'].copy()
        max_actives = archive['max_actives'].copy()
        min_cumul = archive['min_cumul'].copy()
    if campagne != CAMPAGNE:
        raise ValueError('Cache d’une autre campagne.')
    if cle_stockee != canonique(cle):
        raise ValueError('Cache d’une autre configuration.')
    if n_stocke != int(n):
        raise ValueError('Nombre d’échantillons incompatible.')
    if changements.shape != (TOURS, 64):
        raise ValueError('Shape changements invalide.')
    if histogramme.shape != (TOURS, 65):
        raise ValueError('Shape histogramme invalide.')
    if np.any(histogramme.sum(axis=1) != n):
        raise ValueError('Histogramme incompatible avec n.')
    if np.any(min_actives < 1) or np.any(min_actives > 8) or np.any(max_actives < 1) or np.any(max_actives > 8):
        raise ValueError('Nombre de S-box actives invalide.')
    checksum_recalcule = checksum_statistiques(changements, histogramme, somme_actives, min_actives, max_actives, min_cumul)
    if checksum_recalcule != checksum:
        raise ValueError('Empreinte du cache incorrecte.')
    return {'changements': changements, 'histogramme': histogramme, 'somme_actives': somme_actives, 'min_actives': min_actives, 'max_actives': max_actives, 'min_cumul': min_cumul}

def sauver_cache_configuration(chemin, cle, n, statistiques):
    checksum = checksum_statistiques(statistiques['changements'], statistiques['histogramme'], statistiques['somme_actives'], statistiques['min_actives'], statistiques['max_actives'], statistiques['min_cumul'])
    temporaire = chemin.with_name(chemin.name + '.tmp')
    with temporaire.open('wb') as fichier:
        np.savez(fichier, changements=statistiques['changements'], histogramme=statistiques['histogramme'], somme_actives=statistiques['somme_actives'], min_actives=statistiques['min_actives'], max_actives=statistiques['max_actives'], min_cumul=statistiques['min_cumul'], cle=np.asarray(canonique(cle)), campagne=np.asarray(CAMPAGNE), n=np.asarray(int(n)), sha256=np.asarray(checksum))
    os.replace(str(temporaire), str(chemin))

def initialiser():
    global CAMPAGNE, DOSSIER
    if DOSSIER is not None:
        return
    m = RAINHASH_MODULE
    if int(m.T0.cardinality()) != P or int(m.ALPHA_L) != 4 or int(m.NB_TOURS) != TOURS:
        raise ValueError('RainHash doit utiliser p=251, alpha=4 et huit tours.')
    if len(m.CONSTANTES_TOURS) != TOURS:
        raise ValueError('Huit constantes originales sont nécessaires.')
    identite = dict(version='jupyter-lots-v1', source_sha256=empreinte(RAINHASH_SOURCE),
                    test_sha256=empreinte(SOURCE_LOTS),
                    constantes=[repr(c) for c in m.CONSTANTES_TOURS],
                    familles=FAMILLES, deltas=DELTAS, graine=GRAINE)
    campagne = empreinte(canonique(identite))
    dossier = (Path.cwd() / 'resultats_tours_jupyter_alpha4' / campagne).resolve()
    (dossier / 'cache').mkdir(parents=True, exist_ok=True)
    ecrire_json(dossier / 'protocole.json', identite)
    CAMPAGNE, DOSSIER = campagne, dossier

def configurations():
    for famille, n in FAMILLES:
        for position in range(64):
            for delta in DELTAS:
                cle = [famille, int(n), int(position), int(delta)]
                chemin = DOSSIER / 'cache' / ('{}_p{}_d{}.npz'.format(famille, position, delta))
                yield cle, chemin

def statistiques_vides():
    return dict(changements=np.zeros((TOURS,64),dtype=np.int64),
                histogramme=np.zeros((TOURS,65),dtype=np.int64),
                somme_actives=np.zeros(TOURS,dtype=np.int64),
                min_actives=np.full(TOURS,8,dtype=np.int64),
                max_actives=np.zeros(TOURS,dtype=np.int64),
                min_cumul=np.full(TOURS,64,dtype=np.int64))

def lire_configuration(cle, chemin):
    if not chemin.exists():
        return 0, statistiques_vides()
    with np.load(str(chemin), allow_pickle=False) as z:
        fait = int(z['n'].item())
    if not 1 <= fait <= cle[1]:
        raise ValueError('Nombre de paires invalide dans ' + str(chemin))
    return fait, charger_cache_configuration(chemin, cle, fait)

def avancement():
    initialiser()
    fait = 0
    total = 0
    terminees = 0
    for cle, chemin in configurations():
        k, _ = lire_configuration(cle, chemin)
        fait += k
        total += cle[1]
        terminees += (k == cle[1])
    return dict(paires=fait, total=total, configurations_terminees=terminees,
                dossier=str(DOSSIER))

def ETAT_TOURS():
    etat = avancement()
    print('{}/{} paires sauvegardées ({:.2f} %).'.format(
        etat['paires'], etat['total'], 100.0*etat['paires']/etat['total']))
    print('Résultats :', DOSSIER)
    return etat

def VERIFIER_TOURS():
                                                                
    initialiser()
    verifier()
    print('Composition des huit tours et inverse : vérifications réussies.')

def EXECUTER_LOT(max_paires=16, max_secondes=30):

    global EN_COURS
    max_paires = int(max_paires)
    max_secondes = float(max_secondes)
    if max_paires < 1 or not math.isfinite(max_secondes) or max_secondes <= 0:
        raise ValueError('max_paires et max_secondes doivent être strictement positifs.')
    if EN_COURS:
        raise RuntimeError('Un lot est déjà en cours.')
    EN_COURS = True
    afficheur = None
    debut = time.monotonic()
    nouvelles = 0
    depart = None
    try:
        depart = avancement()
        total = depart['total']
        if depart['paires'] == total:
            print('Toutes les paires sont calculées. Exécutez RAPPORT_TOURS().')
            return depart
        afficheur = display('Préparation du lot…', display_id=True)
        for cle, chemin in configurations():
            famille, n, position, delta = cle
            fait, stats = lire_configuration(cle, chemin)
            if fait == n:
                continue
                                                                                
            rng = np.random.default_rng(graine(GRAINE, cle))
            for _ in range(fait):
                generer_etat(rng, famille)
            while fait < n:
                if nouvelles >= max_paires or time.monotonic()-debut >= max_secondes:
                    return None
                def afficher_tour(compteur):
                    tour = int(compteur)
                    ecoule = time.monotonic()-debut
                    vitesse = nouvelles/ecoule if ecoule > 0 else 0
                    restant = ((total-depart['paires']-nouvelles)/vitesse
                               if vitesse else None)
                    texte = ('Calcul | {} | entrée {} | delta {} | paire {}/{} | tour {}/8'
                             '\nLot : {}/{} | total sauvegardé : {}/{} | reste estimé : {}').format(
                        famille, position+1, delta, fait+1, n, tour,
                        nouvelles, max_paires, depart['paires']+nouvelles, total, duree(restant))
                    if afficheur is not None:
                        afficheur.update(texte)
                class Suivi:
                    def avancer(self, compteur):
                        afficher_tour(compteur)
                afficher_tour(0)
                x = generer_etat(rng, famille)
                differences, actives, _ = trace_paire(x, position, delta, Suivi(), 0)

                prochain = {k:v.copy() for k,v in stats.items()}
                masque = differences != 0
                distances = np.count_nonzero(masque, axis=1)
                prochain['changements'] += masque
                prochain['histogramme'][np.arange(TOURS), distances] += 1
                a = actives.astype(np.int64)
                prochain['somme_actives'] += a
                prochain['min_actives'] = np.minimum(prochain['min_actives'], a)
                prochain['max_actives'] = np.maximum(prochain['max_actives'], a)
                prochain['min_cumul'] = np.minimum(prochain['min_cumul'], np.cumsum(a))
                sauver_cache_configuration(chemin, cle, fait+1, prochain)
                stats = prochain
                fait += 1
                nouvelles += 1
                del x, differences, actives, masque, distances, a, prochain

                gc.collect()
    except KeyboardInterrupt:
        print('Lot interrompu. Les paires sauvegardées seront reprises automatiquement.')
    finally:
        EN_COURS = False
        gc.collect()
        if depart is not None:
            etat = avancement()
            texte = ('Pause | {}/{} paires sauvegardées ({:.2f} %) | lot : {} paires'
                     '\nRelancez EXECUTER_LOT() pour continuer.').format(
                         etat['paires'], etat['total'], 100.0*etat['paires']/etat['total'],
                         etat['paires']-depart['paires'])
            if etat['paires'] == etat['total']:
                texte = 'Calcul terminé : {} paires. Exécutez RAPPORT_TOURS().'.format(etat['total'])
            if afficheur is not None:
                afficheur.update(texte)
            else:
                print(texte)

class Progression:
                                                          
    def __init__(self, titre, total, unite=''):
        self.titre = titre
    def __enter__(self):
        return self
    def avancer(self, fait):
        pass
    def __exit__(self, *args):
        return False

def calculer_configuration(famille, n, position, delta, chemin, suivi, compteur):
    cle = [famille, int(n), int(position), int(delta)]
    fait, stats = lire_configuration(cle, chemin)
    if fait != n:
        raise ValueError('Calcul incomplet : relancez EXECUTER_LOT().')
    return stats, compteur

def probabilites_recurrence(egaux, n, p):
    import builtins as B
    import numpy as np
    n = B.int(n)
    p = B.float(p)
    print("1/3 — Construction de la loi binomiale", flush=True)
    masses = [(1.0 - p) ** n]
    rapport = p / (1.0 - p)
    for k in B.range(n):
        suivant = masses[-1] * ((n - k) / (k + 1)) * rapport
        masses.append(suivant)
    print("2/3 — Calcul des deux queues", flush=True)
    gauche = [0.0] * (n + 1)
    droite = [0.0] * (n + 1)
    cumul = 0.0
    for k in B.range(n + 1):
        cumul += masses[k]
        gauche[k] = B.min(1.0, cumul)
    cumul = 0.0
    for k in B.range(n, -1, -1):
        cumul += masses[k]
        droite[k] = B.min(1.0, cumul)
    table = [B.min(1.0, 2.0 * B.min(gauche[k], droite[k])) for k in B.range(n + 1)]
    print("3/3 — Attribution des probabilités aux tests", flush=True)
    comptes = egaux.tolist()
    valeurs = [[[table[B.int(k)] for k in ligne] for ligne in tour] for tour in comptes]
    resultat = np.asarray(valeurs, dtype="float64")
    print("Calcul binomial terminé.", flush=True)
    return resultat

def RAPPORT_TOURS():
    initialiser()
    etat = avancement()
    if etat['paires'] != etat['total']:
        print('Calcul incomplet : {}/{} paires. Relancez EXECUTER_LOT().'.format(
            etat['paires'], etat['total']))
        return None
    debut = perf_counter()
    dossier = DOSSIER
    cache = dossier / 'cache'
    total_paires = etat['total']
    resumes = []
    tests = []
    histogrammes = []
    tables_positions = []
    for numero, (famille, n) in enumerate(FAMILLES, 1):
        configurations = [(position, delta, cache / '{}_p{}_d{}.npz'.format(famille, position, delta)) for position in range(64) for delta in DELTAS]
        manquants = sum((not chemin.exists() for _, _, chemin in configurations))
        print('\n========================================', flush=True)
        print('FAMILLE {}/{} : {}'.format(numero, len(FAMILLES), famille), flush=True)
        print('{} configurations nouvelles à calculer.'.format(manquants), flush=True)
        changements = np.zeros((TOURS, 64, 64), dtype=np.int64)
        histogramme = np.zeros((TOURS, 65), dtype=np.int64)
        somme_actives = np.zeros(TOURS, dtype=np.int64)
        min_actives = np.full(TOURS, 8, dtype=np.int64)
        max_actives = np.zeros(TOURS, dtype=np.int64)
        min_cumul = np.full(TOURS, 64, dtype=np.int64)
        compteur = 0
        with Progression('famille ' + famille, manquants * n * TOURS, 'paires de tours') as suivi:
            for position, delta, chemin in configurations:
                suivi.titre = '{} : entrée {}, delta {}'.format(famille, position + 1, delta)
                stats, compteur = calculer_configuration(famille, n, position, delta, chemin, suivi, compteur)
                changements[:, position, :] += stats['changements']
                histogramme += stats['histogramme']
                somme_actives += stats['somme_actives']
                min_actives = np.minimum(min_actives, stats['min_actives'])
                max_actives = np.maximum(max_actives, stats['max_actives'])
                min_cumul = np.minimum(min_cumul, stats['min_cumul'])
                del stats
                gc.collect()
        echantillons = 64 * len(DELTAS) * n
        par_position = len(DELTAS) * n
        np.save(dossier / ('changements_' + famille + '.npy'), changements, allow_pickle=False)
        valeurs = np.arange(65, dtype=np.float64)
        for r in range(TOURS):
            moyenne = float(histogramme[r] @ valeurs / echantillons)
            variance = float(histogramme[r] @ (valeurs - moyenne) ** 2 / max(1, echantillons - 1))
            support = np.flatnonzero(histogramme[r])
            resumes.append({'famille': famille, 'tour': r + 1, 'paires': echantillons, 'distance_moyenne': moyenne, 'taux_changement_pourcentage': 100.0 * moyenne / 64.0, 'ecart_type_distance': math.sqrt(variance), 'distance_min': int(support[0]), 'distance_max': int(support[-1]), 'Sboxes_actives_moyenne_avant_tour': float(somme_actives[r] / echantillons), 'Sboxes_actives_min_observe_avant_tour': int(min_actives[r]), 'Sboxes_actives_max_observe_avant_tour': int(max_actives[r]), 'Sboxes_actives_cumul_min_observe': int(min_cumul[r])})
            for distance in range(65):
                histogrammes.append({'famille': famille, 'tour': r + 1, 'distance': distance, 'effectif': int(histogramme[r, distance])})
            for position in range(64):
                tables_positions.append({'famille': famille, 'tour': r + 1, 'position_entree': position + 1, 'colonne_K': position // 8 + 1, 'coordonnee_dans_K': position % 8 + 1, 'distance_moyenne': float(changements[r, position].sum() / par_position)})
        if famille == 'uniforme':
            p_egalite = (P ** 63 - 1) / float(P ** 64 - 1)
            egaux = par_position - changements
            pv = probabilites_recurrence(egaux, par_position, p_egalite)
            for r in range(TOURS):
                for position in range(64):
                    for sortie in range(64):
                        tests.append({'tour': r + 1, 'position_entree': position + 1, 'coordonnee_sortie': sortie + 1, 'essais': par_position, 'egalites': int(egaux[r, position, sortie]), 'p_bilaterale': float(pv[r, position, sortie])})
        ecrire_csv(dossier / 'resume_par_tour.csv', resumes)
        ecrire_csv(dossier / 'histogrammes_distances.csv', histogrammes)
        ecrire_csv(dossier / 'diffusion_par_position.csv', tables_positions)
        del changements
        del histogramme
        del somme_actives
        del min_actives
        del max_actives
        del min_cumul
        gc.collect()
    if tests:
        ajustees = correction_holm([r['p_bilaterale'] for r in tests])
        for ligne, p in zip(tests, ajustees):
            ligne['p_holm'] = float(p)
            ligne['rejet_holm'] = bool(p <= ALPHA_GLOBAL)
        ecrire_csv(dossier / 'tests_uniformes_holm.csv', tests)
    conclusion = []
    print('\n========================================')
    print('RÉSULTATS SUR ÉTATS UNIFORMES')
    print('========================================')
    for r in range(1, TOURS + 1):
        groupe = [t for t in tests if t['tour'] == r]
        rejets = sum((t['rejet_holm'] for t in groupe))
        conclusion.append({'tour': r, 'tests': len(groupe), 'rejets_holm': int(rejets)})
        if groupe:
            print('Tour {} : {} rejets / {} tests.'.format(r, rejets, len(groupe)))
    ecrire_json(dossier / 'conclusion.json', {'par_tour': conclusion, 'paires': total_paires})
    print('\nIdéal de référence : environ {:.8f}/64 coordonnées modifiées.'.format(64.0 * 250.0 / 251.0))
    print('RAPPORT TERMINÉ — ' + duree(perf_counter() - debut))
    print('Résultats :', dossier)
    return resumes
'''

import types as _types_tours
                                                                  
if 'RAINHASH_SOURCE' not in globals():
    raise NameError('Exécutez d’abord votre cellule RainHash alpha=4.')
_noms_tours = ('T0', 'ALPHA_L', 'NB_TOURS', 'CONSTANTES_TOURS',
               'L_sans_MDS', 'tour_direct_T3', 'permutation_complete_T3',
               'permutation_inverse_T3')
if all(n in globals() for n in _noms_tours):
    _module_tours = _types_tours.SimpleNamespace(**{n:globals()[n] for n in _noms_tours})
elif 'RAINHASH_MODULE' in globals() and all(hasattr(RAINHASH_MODULE,n) for n in _noms_tours):
    _module_tours = RAINHASH_MODULE
else:
    import sage.all as _sage_tours
    _module_tours = _types_tours.ModuleType('rainhash_jupyter_local')
    _module_tours.__dict__.update({n:getattr(_sage_tours,n) for n in dir(_sage_tours) if not n.startswith('_')})
                                                                 
    exec(compile(str(RAINHASH_SOURCE), '<rainhash_jupyter_local>', 'exec'), _module_tours.__dict__)
    if not all(hasattr(_module_tours,n) for n in _noms_tours):
        raise NameError('Les fonctions RainHash attendues sont absentes de RAINHASH_SOURCE.')
_espace_tours = dict(RAINHASH_MODULE=_module_tours,
                     RAINHASH_SOURCE=str(RAINHASH_SOURCE), SOURCE_LOTS=_SOURCE_TOURS_JUPYTER)
                                                                            
exec(compile(_SOURCE_TOURS_JUPYTER, '<tours_jupyter>', 'exec'), _espace_tours)
EXECUTER_LOT = _espace_tours['EXECUTER_LOT']
ETAT_TOURS = _espace_tours['ETAT_TOURS']
RAPPORT_TOURS = _espace_tours['RAPPORT_TOURS']
VERIFIER_TOURS = _espace_tours['VERIFIER_TOURS']
RAPPORT_TOURS_CORRIGE = RAPPORT_TOURS
print('Prêt. Démarrage du premier lot.')

EXECUTER_LOT(max_paires=16, max_secondes=30)

                  