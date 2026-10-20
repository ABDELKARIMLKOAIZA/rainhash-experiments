from rainhash_sponge_fp251_alpha4_8tours import RAINHASH_MODULE, RAINHASH_SOURCE

TEST_PROJECTIONS_SOURCE = r'''import csv
import hashlib
import json
import math
import os
import threading
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.stats import chi2

P = 251
DIMENSION_SORTIE = 32
LONGUEURS_TESTEES = list(range(1, 17))
NOMBRE_ECHANTILLONS_PAR_LONGUEUR = 50200
COEFFICIENTS_PAIRES = [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, -11]
NOMBRE_PROJECTIONS_POIDS_3 = 256
NOMBRE_PROJECTIONS_POIDS_4 = 256
NOMBRE_PROJECTIONS_DENSES = 256
NOMBRE_CANDIDATS_VALIDATION = 20
NOMBRE_PAIRES_VALIDATION_COMPLETE = 20
NOMBRE_LIGNES_RANG = 1200
ALPHA_GLOBAL = 0.05
SIMULATIONS_MINIMALES = 99999
GRAINE_PROJECTIONS = 2026092911
GRAINE_DETECTION = 2026092912
GRAINE_VALIDATION = 2026092913
GRAINE_SIMULATIONS = 2026092914
GRAINE_PUISSANCE = 2026092915
NOMBRE_REPETITIONS_PUISSANCE = 200
TAILLE_FRAGMENT_CACHE = 1000
INTERVALLE_AFFICHAGE = 10.0
SUIVI_RANG = None

def canonique(objet):
    return json.dumps(objet, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)

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

class Progression:

    def __init__(self, titre, total, unite='unités'):
        self.titre, self.total, self.unite = titre, int(total), unite
        self.fait = 0
        self.debut = self.dernier_progres = perf_counter()
        self.verrou = threading.Lock()
        self.arret = threading.Event()

    def avancer(self, fait):
        with self.verrou:
            if int(fait) > self.fait:
                self.fait = int(fait)
                self.dernier_progres = perf_counter()

    def afficher(self, statut='EN COURS'):
        with self.verrou:
            fait, dernier = self.fait, self.dernier_progres
        ecoule = max(perf_counter() - self.debut, 1e-9)
        vitesse = fait / ecoule
        reste = (self.total - fait) / vitesse if vitesse > 0 else None
        print('[{}] {} | {}/{} {} ({:.1f} %) | écoulé {} | reste étape {} | '
              '{:.2f}/s | dernière progression il y a {:.0f} s'.format(
                  statut, self.titre, fait, self.total, self.unite,
                  100.0 * fait / max(self.total, 1), duree(ecoule),
                  duree(reste), vitesse, perf_counter() - dernier), flush=True)

    def battre(self):
        while not self.arret.wait(INTERVALLE_AFFICHAGE):
            self.afficher()

    def __enter__(self):
        self.afficher()
        self.thread = threading.Thread(target=self.battre, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, genre, erreur, trace):
        self.arret.set()
        self.thread.join(timeout=1.0)
        if genre is None:
            self.avancer(self.total)
            self.afficher('TERMINÉ')
        else:
            self.afficher('INTERROMPU' if genre is KeyboardInterrupt else 'ERREUR')
        return False

def ecrire_json(chemin, objet):
    chemin = Path(chemin)
    temporaire = chemin.with_name(chemin.name + '.tmp')
    temporaire.write_text(json.dumps(objet, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    os.replace(str(temporaire), str(chemin))

def ecrire_csv(chemin, lignes):
    if not lignes:
        return
    chemin = Path(chemin)
    temporaire = chemin.with_name(chemin.name + '.tmp')
    champs = sorted(set().union(*(r.keys() for r in lignes)) - {'projection_obj'})
    with temporaire.open('w', newline='', encoding='utf-8') as fichier:
        writer = csv.DictWriter(fichier, fieldnames=champs)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({k: canonique(ligne[k]) if isinstance(ligne.get(k),
                             (dict, list, tuple)) else ligne.get(k, '') for k in champs})
    os.replace(str(temporaire), str(chemin))

def verifier_hachage():
    module = RAINHASH_MODULE
    attendus = {'ALPHA_L': 4, 'NB_TOURS': 8, 'TAUX_SPONGE': 4,
                'CAPACITE_SPONGE': 4, 'TAILLE_EMPREINTE_SPONGE': 4}
    if int(module.T0.cardinality()) != P:
        raise ValueError('Le corps doit être F_251.')
    for nom, valeur in attendus.items():
        if int(getattr(module, nom)) != valeur:
            raise ValueError('{} doit valoir {}.'.format(nom, valeur))
    for nom in ['hachage_sponge_pi_T3', 'coordonnees_empreinte_F251']:
        if getattr(module, nom).__globals__ is not module.__dict__:
            raise ValueError('Fonction extérieure au module : ' + nom)

    temoins = []
    for longueur in [0, 1, 4, 5, 8]:
        message = [[module.T0((17*i + 31*j + 3) % P) for j in range(8)]
                   for i in range(longueur)]
        temoins.append(convertir_sortie(module.hachage_sponge_pi_T3(message)).tolist())
    return temoins

def convertir_sortie(h):
    valeurs = [int(x) for x in RAINHASH_MODULE.coordonnees_empreinte_F251(h)]
    if len(valeurs) != 32 or any(x < 0 or x >= P for x in valeurs):
        raise ValueError('Sortie invalide : 32 coordonnées dans [0, 250] attendues.')
    return np.asarray(valeurs, dtype=np.int64)

def obtenir_sorties(longueur, phase):
    n = NOMBRE_ECHANTILLONS_PAR_LONGUEUR
    resultat = np.empty((n, 32), dtype=np.int64)
    base = GRAINE_DETECTION if phase == 'détection' else GRAINE_VALIDATION
    titre = '{} : hachage L={}'.format(phase, longueur)
    with Progression(titre, n, 'messages') as suivi:
        for debut in range(0, n, TAILLE_FRAGMENT_CACHE):
            fin = min(n, debut + TAILLE_FRAGMENT_CACHE)
            cle = empreinte(canonique([phase, longueur, debut, fin]))
            chemin = CACHE / ('sorties_' + cle + '.npz')
            if chemin.exists():
                try:
                    with np.load(str(chemin), allow_pickle=False) as archive:
                        bloc = archive['sorties'].copy()
                        checksum = str(archive['sha256'].item())
                        campagne = str(archive['campagne'].item())
                    if (campagne != CAMPAGNE or bloc.dtype != np.uint8
                        or bloc.shape != (fin - debut, 32) or np.any(bloc >= P)
                        or hashlib.sha256(bloc.tobytes()).hexdigest() != checksum):
                        raise ValueError('Métadonnées, type, dimensions ou intégrité invalides.')
                except Exception as erreur:
                    raise RuntimeError('Cache invalide : {}. Renommez ce fragment '
                                       'pour le recalculer.'.format(chemin)) from erreur
            else:
                rng = np.random.default_rng(graine(base, longueur, debut))
                bloc = np.empty((fin - debut, 32), dtype=np.uint8)
                for k in range(fin - debut):
                    valeurs = rng.integers(0, P, size=(longueur, 8), dtype=np.int64)
                    message = [[RAINHASH_MODULE.T0(int(x)) for x in ligne]
                               for ligne in valeurs]
                    bloc[k] = convertir_sortie(RAINHASH_MODULE.hachage_sponge_pi_T3(message))
                    suivi.avancer(debut + k + 1)
                temporaire = chemin.with_name(chemin.name + '.tmp')
                with temporaire.open('wb') as fichier:
                    np.savez_compressed(fichier, sorties=bloc,
                        campagne=np.asarray(CAMPAGNE),
                        sha256=np.asarray(hashlib.sha256(bloc.tobytes()).hexdigest()))
                os.replace(str(temporaire), str(chemin))
            resultat[debut:fin] = bloc
            suivi.avancer(fin)
    return resultat

def normaliser_projection(indices, coefficients):
    indices = tuple(int(i) for i in indices)
    coefficients = [int(c) % P for c in coefficients]
    if len(indices) != len(coefficients) or len(indices) == 0:
        raise ValueError("Projection mal definie.")
    if len(set(indices)) != len(indices):
        raise ValueError("Les indices d'une projection doivent etre distincts.")
    if any(i < 0 or i >= DIMENSION_SORTIE for i in indices):
        raise ValueError("Indice de projection invalide.")
    if any(c == 0 for c in coefficients):
        raise ValueError("Les coefficients du support doivent etre non nuls.")
    inverse = pow(int(coefficients[0]), P - 2, P)
    coefficients = tuple((c * inverse) % P for c in coefficients)
    return indices, coefficients

def supports_equilibres(poids, nombre, generateur):
    compteurs = np.zeros(DIMENSION_SORTIE, dtype=np.int64)
    supports = []
    deja = set()
    tentatives = 0
    while len(supports) < nombre:
        bruit = generateur.random(DIMENSION_SORTIE)
        ordre = np.lexsort((bruit, compteurs))
        bassin = ordre[:min(DIMENSION_SORTIE, max(2 * poids, poids + 3))]
        support = tuple(sorted(int(x) for x in generateur.choice(
            bassin, size=poids, replace=False,
        )))
        tentatives += 1
        if support in deja:
            if tentatives > 10000:
                raise RuntimeError("Impossible de construire des supports distincts.")
            continue
        deja.add(support)
        supports.append(support)
        compteurs[list(support)] += 1
    return supports, compteurs

def construire_projections():
    generateur = np.random.default_rng(int(GRAINE_PROJECTIONS))
    projections = []
    cles = set()

    def ajouter(famille, nom, indices, coefficients):
        indices, coefficients = normaliser_projection(indices, coefficients)
        cle = (indices, coefficients)
        if cle in cles:
            return False
        cles.add(cle)
        projections.append({
            "id_projection": len(projections) + 1,
            "famille": str(famille),
            "nom": str(nom),
            "indices": indices,
            "coefficients": coefficients,
            "poids": len(indices),
        })
        return True

    for i in range(DIMENSION_SORTIE):
        ajouter("coordonnee", "H_{}".format(i + 1), [i], [1])

    for i, j in combinations(range(DIMENSION_SORTIE), 2):
        for a in COEFFICIENTS_PAIRES:
            a_mod = int(a) % P
            ajouter(
                "paire",
                "H_{}{}{}H_{}".format(
                    i + 1, "+" if int(a) >= 0 else "-", abs(int(a)), j + 1,
                ),
                [i, j], [1, a_mod],
            )

    for poids, nombre, famille in [
        (3, NOMBRE_PROJECTIONS_POIDS_3, "poids_3"),
        (4, NOMBRE_PROJECTIONS_POIDS_4, "poids_4"),
    ]:
        supports, compteurs = supports_equilibres(poids, nombre, generateur)
        for numero, support in enumerate(supports, start=1):
            while True:
                coefficients = generateur.integers(1, P, size=poids, dtype=np.int64)
                if ajouter(
                    famille, "{}_{}".format(famille, numero),
                    support, coefficients,
                ):
                    break
        print(
            "Equilibrage {} : apparitions min={}, max={}".format(
                famille, int(np.min(compteurs)), int(np.max(compteurs)),
            )
        )

    nombre_dense = 0
    while nombre_dense < NOMBRE_PROJECTIONS_DENSES:
        coefficients = generateur.integers(
            1, P, size=DIMENSION_SORTIE, dtype=np.int64,
        )
        if ajouter(
            "dense", "dense_{}".format(nombre_dense + 1),
            range(DIMENSION_SORTIE), coefficients,
        ):
            nombre_dense += 1

    return projections

def evaluer_projection(sorties, projection):
    indices = np.asarray(projection["indices"], dtype=np.int64)
    coefficients = np.asarray(projection["coefficients"], dtype=np.int64)
    if len(indices) == 1:
        return (sorties[:, indices[0]] * coefficients[0]) % P
    valeurs = np.zeros(sorties.shape[0], dtype=np.int64)
    for indice, coefficient in zip(indices, coefficients):
        valeurs += sorties[:, int(indice)] * int(coefficient)
    return valeurs % P

def statistiques_frequences(frequences):
    frequences = np.asarray(frequences, dtype=np.int64)
    if frequences.shape != (P,):
        raise ValueError("Il faut exactement 251 frequences.")
    n = int(np.sum(frequences))
    if n <= 0:
        raise ValueError("Distribution vide.")
    attendu = float(n) / P
    ecarts_comptes = frequences.astype(float) - attendu
    statistique = float(np.sum(ecarts_comptes ** 2 / attendu))
    p_superieure = float(chi2.sf(statistique, P - 1))
    p_inferieure = float(chi2.cdf(statistique, P - 1))
    p_deux_cotes = min(1.0, 2.0 * min(p_superieure, p_inferieure))
    probabilites = frequences.astype(float) / float(n)
    uniforme = 1.0 / P
    ecarts = np.abs(probabilites - uniforme)
    maximum = float(np.max(probabilites))
    spectre = np.fft.fft(probabilites)
    biais_fourier = float(np.max(np.abs(spectre[1:])))
    valeur_max = int(np.argmax(frequences))
    return {
        "n": n,
        "chi2": statistique,
        "p_superieure": p_superieure,
        "p_inferieure": p_inferieure,
        "p_deux_cotes": p_deux_cotes,
        "frequence_min": int(np.min(frequences)),
        "frequence_max": int(np.max(frequences)),
        "valeur_frequence_max": valeur_max,
        "biais_maximal": float(np.max(ecarts)),
        "variation_totale": float(0.5 * np.sum(ecarts)),
        "min_entropie_empirique_descriptive": float(
            -math.log(maximum, 2)
        ),
        "biais_fourier_max": biais_fourier,
    }

def analyser_projection(sorties, longueur, projection, phase):
    valeurs = evaluer_projection(sorties, projection)
    frequences = np.bincount(valeurs, minlength=P)
    resultat = statistiques_frequences(frequences)
    resultat.update({
        "phase": str(phase),
        "longueur": int(longueur),
        "id_projection": int(projection["id_projection"]),
        "famille": projection["famille"],
        "nom": projection["nom"],
        "poids": int(projection["poids"]),
        "indices": ";".join(str(int(i) + 1) for i in projection["indices"]),
        "coefficients": ";".join(str(int(c)) for c in projection["coefficients"]),
        "projection_obj": projection,
    })
    return resultat

def correction_holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.asarray([], dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values invalides")
    ordre = np.argsort(p_values)
    ajustees = np.empty(m, dtype=float)
    maximum = 0.0
    for rang, indice in enumerate(ordre):
        valeur = min(1.0, (m - rang) * float(p_values[indice]))
        maximum = max(maximum, valeur)
        ajustees[indice] = maximum
    return ajustees

def forme_echelon_modulaire(matrice, p=P):
    a = np.asarray(matrice, dtype=np.int64).copy() % int(p)
    lignes, colonnes = a.shape
    rang = 0
    colonnes_pivots = []
    for colonne in range(colonnes):
        if SUIVI_RANG is not None:
            SUIVI_RANG.avancer(colonne)
        candidats = np.flatnonzero(a[rang:, colonne])
        if candidats.size == 0:
            continue
        pivot = rang + int(candidats[0])
        if pivot != rang:
            a[[rang, pivot]] = a[[pivot, rang]]
        inverse = pow(int(a[rang, colonne]), int(p) - 2, int(p))
        a[rang] = (a[rang] * inverse) % int(p)
        if rang + 1 < lignes:
            facteurs = a[rang + 1:, colonne].copy()
            non_nuls = np.flatnonzero(facteurs)
            if non_nuls.size:
                lignes_cibles = rang + 1 + non_nuls
                a[lignes_cibles] = (
                    a[lignes_cibles]
                    - facteurs[non_nuls, None] * a[rang][None, :]
                ) % int(p)
        rang += 1
        colonnes_pivots.append(int(colonne))
        if rang == lignes:
            break
    return a[:rang], colonnes_pivots

def vecteur_noyau_modulaire(matrice, p=P):

    echelon, colonnes_pivots = forme_echelon_modulaire(
        matrice,
        p=p,
    )
    nombre_colonnes = int(np.asarray(matrice).shape[1])
    pivots = set(colonnes_pivots)
    colonnes_libres = [
        colonne for colonne in range(nombre_colonnes)
        if colonne not in pivots
    ]
    if not colonnes_libres:
        return None

    vecteur = np.zeros(nombre_colonnes, dtype=np.int64)
    vecteur[colonnes_libres[0]] = 1

    for ligne in range(len(colonnes_pivots) - 1, -1, -1):
        pivot = int(colonnes_pivots[ligne])
        somme = int(
            np.dot(
                echelon[ligne, pivot + 1:],
                vecteur[pivot + 1:],
            )
            % int(p)
        )
        vecteur[pivot] = (-somme) % int(p)

    if not np.any(vecteur % int(p)):
        raise AssertionError("Le vecteur du noyau est nul.")
    if np.any((np.asarray(matrice, dtype=np.int64) @ vecteur) % int(p)):
        raise AssertionError("Le vecteur calcule n'appartient pas au noyau.")
    return vecteur % int(p)

def matrice_monomes_degre_2(sorties):
    x = np.asarray(sorties, dtype=np.int64) % P
    n = x.shape[0]
    i, j = np.triu_indices(DIMENSION_SORTIE, k=1)
    return np.concatenate(
        [
            np.ones((n, 1), dtype=np.int64),
            x,
            (x * x) % P,
            (x[:, i] * x[:, j]) % P,
        ],
        axis=1,
    ) % P

def evaluer_relation_quadratique(sorties, coefficients):
    x = np.asarray(sorties, dtype=np.int64) % P
    coefficients = np.asarray(coefficients, dtype=np.int64) % P
    nombre_monomes = 1 + 2 * DIMENSION_SORTIE + (
        DIMENSION_SORTIE * (DIMENSION_SORTIE - 1) // 2
    )
    if coefficients.shape != (nombre_monomes,):
        raise ValueError("Dimension incorrecte pour la relation quadratique.")

    position = 0
    valeurs = np.full(x.shape[0], int(coefficients[position]), dtype=np.int64)
    position += 1
    valeurs += x @ coefficients[position:position + DIMENSION_SORTIE]
    position += DIMENSION_SORTIE
    valeurs += (x * x) @ coefficients[position:position + DIMENSION_SORTIE]
    position += DIMENSION_SORTIE

    i, j = np.triu_indices(DIMENSION_SORTIE, k=1)
    for indice_i, indice_j, coefficient in zip(
        i,
        j,
        coefficients[position:],
    ):
        if int(coefficient) != 0:
            valeurs += (
                int(coefficient)
                * x[:, int(indice_i)]
                * x[:, int(indice_j)]
            )

    return valeurs % P

def analyser_liste(sorties, longueur, projections, phase):
    resultats = []
    with Progression('{} : projections L={}'.format(phase, longueur),
                     len(projections), 'projections') as suivi:
        for k, projection in enumerate(projections, 1):
            resultat = analyser_projection(sorties, longueur, projection, phase)
            comptes = np.bincount(evaluer_projection(sorties, projection), minlength=P)

            resultat['somme_carres_comptes'] = int(np.sum(comptes * comptes))
            resultats.append(resultat)
            suivi.avancer(k)
    return resultats

def analyser_relations(sorties, longueur):
    global SUIVI_RANG
    lignes = min(NOMBRE_LIGNES_RANG, len(sorties))
    if lignes < 561:
        raise ValueError('Au moins 561 lignes pour chercher les relations quadratiques.')
    x = sorties[:lignes]
    resultats = []
    matrices = [('affine', np.column_stack((np.ones(lignes, dtype=np.int64), x))),
                ('quadratique', matrice_monomes_degre_2(x))]
    for nature, matrice in matrices:
        titre = 'détection : rang {} L={}'.format(nature, longueur)
        with Progression(titre, matrice.shape[1], 'colonnes') as suivi:
            SUIVI_RANG = suivi
            try:
                echelon, pivots = forme_echelon_modulaire(matrice)
            finally:
                SUIVI_RANG = None
        libres = [j for j in range(matrice.shape[1]) if j not in set(pivots)]
        relation = None
        if libres:

            relation = np.zeros(matrice.shape[1], dtype=np.int64)
            relation[libres[0]] = 1
            for i in range(len(pivots) - 1, -1, -1):
                j = pivots[i]
                relation[j] = -int(echelon[i, j+1:] @ relation[j+1:]) % P
            if not np.any(relation) or np.any((matrice @ relation) % P):
                raise AssertionError('Relation de noyau incorrecte.')
        resultats.append({
            'longueur': longueur, 'nature': nature, 'lignes_detection': lignes,
            'rang': len(pivots), 'colonnes': matrice.shape[1],
            'dimension_noyau': len(libres),
            'coefficients': None if relation is None else relation.tolist(),
            'portee': 'rang sur le sous-echantillon ; une relation retenue si noyau non nul',
        })
    return resultats

def selectionner_plan(detection, projections):
    ajustees = correction_holm([r['p_deux_cotes'] for r in detection])
    for r, p in zip(detection, ajustees):
        r['p_holm_exploratoire_asymptotique'] = float(p)
    classement = sorted(detection, key=lambda r: (r['p_deux_cotes'],
                        r['longueur'], r['id_projection']))
    signaux = [r for r in classement if r['p_holm_exploratoire_asymptotique'] <= ALPHA_GLOBAL]
    candidats = list(signaux)
    presents = {(r['longueur'], r['id_projection']) for r in candidats}
    for r in classement:
        if len(candidats) >= max(NOMBRE_CANDIDATS_VALIDATION, len(signaux)):
            break
        if (r['longueur'], r['id_projection']) not in presents:
            candidats.append(r)
            presents.add((r['longueur'], r['id_projection']))

    paires = []
    deja = set()
    for r in classement:
        p = r['projection_obj']
        if p['famille'] != 'paire':
            continue
        cle = (r['longueur'], tuple(p['indices']))
        if cle not in deja:
            paires.append(cle)
            deja.add(cle)
        if len(paires) == NOMBRE_PAIRES_VALIDATION_COMPLETE:
            break

    plan = {}
    def ajouter(longueur, projection, origine):
        indices, coefficients = normaliser_projection(projection['indices'],
                                                       projection['coefficients'])
        cle = (longueur, indices, coefficients)
        if cle not in plan:
            plan[cle] = {'longueur': longueur, 'projection': projection, 'origines': []}
        if origine not in plan[cle]['origines']:
            plan[cle]['origines'].append(origine)

    for r in candidats:
        ajouter(r['longueur'], r['projection_obj'], 'candidat')
    identifiant = 10000000
    for longueur, (i, j) in paires:
        for a in range(P + 1):
            identifiant += 1
            indices = [j] if a == P else ([i] if a == 0 else [i, j])
            coefficients = [1] if a in (0, P) else [1, a]
            projection = {'id_projection': identifiant, 'famille': 'scan_252',
                'nom': 'H_{}'.format(j+1) if a == P else 'H_{}+{}H_{}'.format(i+1, a, j+1),
                'indices': indices, 'coefficients': coefficients, 'poids': len(indices)}
            ajouter(longueur, projection, 'scan_paire_{}_{}'.format(i+1, j+1))

    return list(plan.values()), len(candidats), paires

def calibrer_monte_carlo(nombre_tests):

    b = max(SIMULATIONS_MINIMALES, int(math.ceil(4 * nombre_tests / ALPHA_GLOBAL)))
    n = NOMBRE_ECHANTILLONS_PAR_LONGUEUR
    chemin = CACHE / ('calibration_MC_{}.npz'.format(b))
    if chemin.exists():
        with np.load(str(chemin), allow_pickle=False) as archive:
            valeurs = archive['valeurs'].copy()
            checksum = str(archive['sha256'].item())
        if (valeurs.dtype != np.int64 or valeurs.shape != (b,)
            or np.any(valeurs[1:] < valeurs[:-1])
            or hashlib.sha256(valeurs.tobytes()).hexdigest() != checksum):
            raise ValueError('Cache Monte-Carlo invalide : ' + str(chemin))
        print('Calibration Monte-Carlo reprise du cache.', flush=True)
        return valeurs
    rng = np.random.default_rng(GRAINE_SIMULATIONS)
    valeurs = np.empty(b, dtype=np.int64)
    with Progression('calibration Monte-Carlo', b, 'simulations') as suivi:
        for debut in range(0, b, 512):
            fin = min(b, debut+512)
            tables = rng.multinomial(n, np.full(P, 1.0/P), size=fin-debut)
            valeurs[debut:fin] = np.sum(tables * tables, axis=1)
            suivi.avancer(fin)
        valeurs.sort()
    temporaire = chemin.with_name(chemin.name + '.tmp')
    with temporaire.open('wb') as fichier:
        np.savez_compressed(fichier, valeurs=valeurs,
            sha256=np.asarray(hashlib.sha256(valeurs.tobytes()).hexdigest()))
    os.replace(str(temporaire), str(chemin))
    return valeurs

def p_mc(observe, nulles):

    b = len(nulles)
    inf = int(np.searchsorted(nulles, observe, side='right'))
    sup = b - int(np.searchsorted(nulles, observe, side='left'))
    return min(1.0, 2.0 * min((inf+1.0)/(b+1.0), (sup+1.0)/(b+1.0)))

def confirmer_relation(sorties, relation):
    coefficients = np.asarray(relation['coefficients'], dtype=np.int64)
    zeros = 0
    with Progression('validation : relation {} L={}'.format(
            relation['nature'], relation['longueur']), len(sorties), 'sorties') as suivi:
        for debut in range(0, len(sorties), 2000):
            x = sorties[debut:debut+2000]
            if relation['nature'] == 'affine':
                valeurs = (coefficients[0] + x @ coefficients[1:]) % P
            else:
                valeurs = evaluer_relation_quadratique(x, coefficients)
            zeros += int(np.sum(valeurs == 0))
            suivi.avancer(min(len(sorties), debut+2000))
    return dict(relation, zeros_validation=zeros, n_validation=len(sorties),
                verifiee_sur_tout_echantillon_validation=(zeros == len(sorties)),
                portee='relation fixee en detection et retestee sur un echantillon independant')

def controle_sensibilite(nombre_tests, nulles):

    rng = np.random.default_rng(GRAINE_PUISSANCE)
    n = NOMBRE_ECHANTILLONS_PAR_LONGUEUR
    lignes = []
    proportions = [0.0, 0.001, 0.005, 0.01, 1.0]
    with Progression('sensibilité : projection connue',
            len(proportions)*NOMBRE_REPETITIONS_PUISSANCE, 'répétitions') as suivi:
        fait = 0
        for proportion in proportions:

            probabilites = np.full(P, (1.0-proportion)/P)
            probabilites[0] += proportion
            rejets = 0
            for _ in range(NOMBRE_REPETITIONS_PUISSANCE):
                comptes = rng.multinomial(n, probabilites)
                valeur = p_mc(int(np.sum(comptes*comptes)), nulles)
                rejets += int(valeur <= ALPHA_GLOBAL/nombre_tests)
                fait += 1
                suivi.avancer(fait)
            lignes.append({'proportion_injection': proportion,
                'repetitions': NOMBRE_REPETITIONS_PUISSANCE,
                'seuil_individuel': ALPHA_GLOBAL/nombre_tests,
                'taux_detection': rejets/float(NOMBRE_REPETITIONS_PUISSANCE),
                'portee': 'sensibilite ciblee : projection connue, seuil Bonferroni'})
    return lignes

def executer():
    global CAMPAGNE, DOSSIER, CACHE
    debut_total = perf_counter()
    with Progression('initialisation : identification du hachage alpha=4', 1) as suivi:
        temoins = verifier_hachage()
        suivi.avancer(1)
    identite = {'hachage_sha256': empreinte(RAINHASH_SOURCE),
               'test_sha256': empreinte(TEST_PROJECTIONS_SOURCE),
               'temoins': temoins, 'numpy': np.__version__, 'scipy': scipy.__version__}
    CAMPAGNE = empreinte(canonique(identite))
    DOSSIER = Path('resultats_projections_alpha4') / CAMPAGNE
    CACHE = DOSSIER / 'cache'
    CACHE.mkdir(parents=True, exist_ok=True)
    ecrire_json(DOSSIER/'identite.json', identite)
    (DOSSIER/'hachage_source.py').write_text(RAINHASH_SOURCE, encoding='utf-8')
    (DOSSIER/'test_source.py').write_text(TEST_PROJECTIONS_SOURCE, encoding='utf-8')
    projections = construire_projections()
    ecrire_json(DOSSIER/'projections.json', projections)
    print('Sponge : p=251, alpha=4, huit tours, r=c=4.\n'
          '{} projections par longueur ; {} tests exploratoires ; {} hachages de détection.'
          .format(len(projections), len(projections)*len(LONGUEURS_TESTEES),
                  len(LONGUEURS_TESTEES)*NOMBRE_ECHANTILLONS_PAR_LONGUEUR), flush=True)
    print('Le temps restant concerne l’étape affichée. Le coût de validation dépend '
          'de la sélection. Ctrl+C interrompt ; relancer reprend les fragments sauvegardés.', flush=True)

    detection, relations = [], []
    for numero, longueur in enumerate(LONGUEURS_TESTEES, 1):
        print('\nPHASE 1/4 — Détection : longueur {}/{} (L={})'.format(
            numero, len(LONGUEURS_TESTEES), longueur), flush=True)
        chemin = CACHE/('analyses_detection_L{}.json'.format(longueur))
        if chemin.exists():
            contenu = json.loads(chemin.read_text(encoding='utf-8'))
            detection.extend(contenu['projections'])
            relations.extend(contenu['relations'])
            print('Analyses de cette longueur reprises du cache.', flush=True)
            continue
        sorties = obtenir_sorties(longueur, 'détection')
        lignes = analyser_liste(sorties, longueur, projections, 'détection')
        rangs = analyser_relations(sorties, longueur)
        ecrire_json(chemin, {'projections': lignes, 'relations': rangs})
        detection.extend(lignes)
        relations.extend(rangs)

    plan, nombre_candidats, paires = selectionner_plan(detection, projections)
    critiques = [r for r in relations if r['coefficients'] is not None]
    ecrire_csv(DOSSIER/'detection_exploratoire.csv', detection)
    ecrire_json(DOSSIER/'rangs_detection.json', relations)
    ecrire_json(DOSSIER/'plan_avant_validation.json', {
        'tests': plan, 'paires': paires, 'relations': critiques,
        'seuil_holm_unique': ALPHA_GLOBAL})
    print('\nPHASE 2/4 — Calibration ; {} tests confirmatoires distincts.'.format(len(plan)), flush=True)
    nulles = calibrer_monte_carlo(len(plan))
    validation, relations_validation = [], []
    longueurs = sorted({r['longueur'] for r in plan} | {r['longueur'] for r in critiques})
    for numero, longueur in enumerate(longueurs, 1):
        print('\nPHASE 3/4 — Validation : longueur {}/{} (L={})'.format(
            numero, len(longueurs), longueur), flush=True)
        sorties = obtenir_sorties(longueur, 'validation')
        sous_plan = [r for r in plan if r['longueur'] == longueur]
        lignes = analyser_liste(sorties, longueur,
                    [r['projection'] for r in sous_plan], 'validation')
        for r, specification in zip(lignes, sous_plan):
            r['origines'] = specification['origines']
            r['p_mc_bilaterale'] = p_mc(r['somme_carres_comptes'], nulles)
        validation.extend(lignes)
        for relation in critiques:
            if relation['longueur'] == longueur:
                relations_validation.append(confirmer_relation(sorties, relation))
        ecrire_json(DOSSIER/'validation_partielle.json', validation)
        ecrire_json(DOSSIER/'relations_validation.json', relations_validation)

    ajustees = correction_holm([r['p_mc_bilaterale'] for r in validation])
    for r, p in zip(validation, ajustees):
        r['p_mc_holm'] = float(p)
        r['rejet_holm'] = bool(p <= ALPHA_GLOBAL)
    ecrire_csv(DOSSIER/'validation_confirmatoire.csv', validation)
    ecrire_json(DOSSIER/'validation_confirmatoire.json', validation)
    print('\nPHASE 4/4 — Sensibilité à une relation synthétique connue.', flush=True)
    sensibilite = controle_sensibilite(len(plan), nulles)
    ecrire_csv(DOSSIER/'sensibilite_projection_connue.csv', sensibilite)
    rejets = sum(r['rejet_holm'] for r in validation)
    relations_reproduites = sum(r['verifiee_sur_tout_echantillon_validation']
                               for r in relations_validation)
    resume = {
        'alpha_couche_lineaire': 4, 'tours': 8,
        'longueurs_en_elements_K': LONGUEURS_TESTEES,
        'n_par_longueur_et_phase': NOMBRE_ECHANTILLONS_PAR_LONGUEUR,
        'projections_par_longueur': len(projections),
        'tests_exploratoires': len(detection), 'candidats_retenus': nombre_candidats,
        'paires_scan_252': len(paires), 'tests_confirmatoires_distincts': len(validation),
        'simulations_MC_communes': len(nulles), 'rejets_MC_Holm': rejets,
        'seuil_global_validation': ALPHA_GLOBAL,
        'relations_reproduites_sur_echantillon_independant': relations_reproduites,
        'duree_secondes': perf_counter()-debut_total,
        'conclusion': ('Des écarts sont détectés dans la famille confirmatoire.' if rejets else
          'Aucun écart significatif dans la famille sélectionnée puis validée.'),
        'portee_rangs': 'Rangs des matrices d’évaluation sur les échantillons étudiés.',
        'portee_MC': 'Modèle nul de sorties indépendantes uniformes ; pool nul commun, '
                    'indépendant des sorties. Holm accepte la dépendance entre p-values.',
        'portee_entropie': 'Min-entropie empirique descriptive.'}
    ecrire_json(DOSSIER/'resume.json', resume)
    print('\nCAMPAGNE TERMINÉE — ' + duree(perf_counter()-debut_total), flush=True)
    print(resume['conclusion'])
    print('Rejets MC + Holm :', rejets)
    print('Relations reproduites sur l’échantillon indépendant :', relations_reproduites)
    print('Résultats :', DOSSIER)
    return resume

RESUME_PROJECTIONS = executer()
'''

if 'RAINHASH_MODULE' not in globals() or 'RAINHASH_SOURCE' not in globals():
    raise NameError('Exécutez d’abord la cellule 1 de hachage alpha=4.')

_espace_projections = {
    '__name__': 'test_projections_alpha4',
    'RAINHASH_MODULE': RAINHASH_MODULE,
    'RAINHASH_SOURCE': RAINHASH_SOURCE,
    'TEST_PROJECTIONS_SOURCE': TEST_PROJECTIONS_SOURCE,
}
exec(compile(TEST_PROJECTIONS_SOURCE, '<test_projections_alpha4>', 'exec'),
     _espace_projections)
RESUME_PROJECTIONS = _espace_projections['RESUME_PROJECTIONS']
