from rainhash_sponge_fp251_alpha4_8tours import RAINHASH_MODULE, RAINHASH_SOURCE

TEST_DIFFERENTIEL_SOURCE = r'''import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import chi2

P = 251
DIM_K = 8
DIM_SORTIE = 32

LONGUEURS = [1, 3, 4, 5, 7, 8, 9, 16]
DELTAS = [1, 2, 127]

N_DETECTION = 2510
N_VALIDATION = 50200

N_STRUCTUREES_PAR_LONGUEUR = 4
N_ORDRE2_PAR_LONGUEUR = 4
N_PROJECTIONS = 16
N_CANDIDATS = 20

B_MONTE_CARLO = 9999
SEUIL_GLOBAL = 0.05
TAILLE_LOT = 16

GRAINE_PLAN = 2026092901
GRAINE_DETECTION = 2026092902
GRAINE_VALIDATION = 2026092903
GRAINE_MONTE_CARLO = 2026092904

module = RAINHASH_MODULE
T0 = module.T0
hachage = module.hachage_sponge_pi_T3
aplatir = module.coordonnees_empreinte_F251

parametres_attendus = {
    "ALPHA_L": 4,
    "NB_TOURS": 8,
    "TAUX_SPONGE": 4,
    "CAPACITE_SPONGE": 4,
    "TAILLE_EMPREINTE_SPONGE": 4,
}

if int(T0.cardinality()) != P:
    raise ValueError("Le hachage doit utiliser F_251.")

for nom, valeur in parametres_attendus.items():
    if int(getattr(module, nom)) != valeur:
        raise ValueError(
            "{} doit valoir {}.".format(nom, valeur)
        )

if hachage.__globals__ is not module.__dict__:
    raise ValueError("La fonction de hachage ne provient pas du module attendu.")

if aplatir.__globals__ is not module.__dict__:
    raise ValueError("La fonction de conversion ne provient pas du module attendu.")

def sha256(texte):
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()

def canonique(objet):
    return json.dumps(
        objet, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False
    )

def graine(*objets):
    empreinte = hashlib.sha256(
        canonique(objets).encode("utf-8")
    ).digest()
    return int.from_bytes(empreinte[:16], "big")

def ecrire_json(chemin, objet):
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_text(
        json.dumps(
            objet, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8"
    )
    os.replace(str(temporaire), str(chemin))

def lire_json(chemin):
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except Exception as erreur:
        raise RuntimeError(
            "Cache illisible : {}. Renommez ce fichier pour le recalculer."
            .format(chemin)
        ) from erreur

def ecrire_csv(chemin, lignes):
    if not lignes:
        return
    temporaire = chemin.with_name(chemin.name + ".tmp")
    champs = sorted(set().union(*(ligne.keys() for ligne in lignes)))
    with temporaire.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=champs)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({
                cle: canonique(valeur)
                if isinstance(valeur, (list, dict)) else valeur
                for cle, valeur in ligne.items()
            })
    os.replace(str(temporaire), str(chemin))

HASH_SOURCE_SHA256 = sha256(RAINHASH_SOURCE)
TEST_SOURCE_SHA256 = sha256(TEST_DIFFERENTIEL_SOURCE)

IDENTIFIANT = sha256(canonique({
    "hachage": HASH_SOURCE_SHA256,
    "test": TEST_SOURCE_SHA256,
}))

DOSSIER = Path("resultats_differentiel_alpha4") / IDENTIFIANT
CACHE = DOSSIER / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def sortie(message):

    coordonnees = aplatir(hachage([
        [T0(int(x)) for x in ligne]
        for ligne in message
    ]))

    if len(coordonnees) != DIM_SORTIE:
        raise ValueError("L'empreinte doit avoir 32 coordonnées.")

    valeurs = [int(x) for x in coordonnees]
    if any(x < 0 or x >= P for x in valeurs):
        raise ValueError("Coordonnée de sortie hors de [0, 250].")

    return np.asarray(valeurs, dtype=np.int64)

def modifier(message, modifications):
    resultat = message.copy()
    for position, delta in modifications:
        i, j = divmod(int(position), DIM_K)
        resultat[i, j] = (int(resultat[i, j]) + int(delta)) % P
    return resultat

def difference(message, configuration, h0):
    d1 = configuration["delta1"]
    h1 = sortie(modifier(message, d1))

    if configuration["ordre"] == 1:
        return (h1 - h0) % P

    d2 = configuration["delta2"]
    h2 = sortie(modifier(message, d2))
    h12 = sortie(modifier(modifier(message, d1), d2))
    return (h12 - h1 - h2 + h0) % P

def creer_configuration(longueur, d1, d2=None):
    cfg = {
        "longueur": int(longueur),
        "ordre": 1 if d2 is None else 2,
        "delta1": sorted([[int(i), int(d)] for i, d in d1]),
        "delta2": [] if d2 is None else sorted([
            [int(i), int(d)] for i, d in d2
        ]),
    }
    cfg["id"] = sha256(canonique(cfg))
    return cfg

def construire_configurations():
    rng = np.random.default_rng(GRAINE_PLAN)
    configurations = {}

    def ajouter(cfg):
        configurations[cfg["id"]] = cfg

    for longueur in LONGUEURS:
        dimension = DIM_K * longueur

        for position in range(dimension):
            for delta in DELTAS:
                ajouter(creer_configuration(
                    longueur, [(position, delta)]
                ))

        for numero in range(N_STRUCTUREES_PAR_LONGUEUR):
            poids = 2 + numero % 3
            support = rng.choice(dimension, size=poids, replace=False)
            d1 = [
                (int(position), int(rng.choice(DELTAS)))
                for position in support
            ]
            ajouter(creer_configuration(longueur, d1))

        for _ in range(N_ORDRE2_PAR_LONGUEUR):
            i, j = rng.choice(dimension, size=2, replace=False)
            ajouter(creer_configuration(
                longueur,
                [(int(i), int(rng.choice(DELTAS)))],
                [(int(j), int(rng.choice(DELTAS)))]
            ))

    return sorted(
        configurations.values(),
        key=lambda cfg: (cfg["longueur"], cfg["ordre"], cfg["id"])
    )

def construire_projections():
    rng = np.random.default_rng(graine(GRAINE_PLAN, "projections"))
    projections = []
    deja = set()
    utilisations = np.zeros(DIM_SORTIE, dtype=np.int64)

    while len(projections) < N_PROJECTIONS:
        poids = [2, 3, 4, DIM_SORTIE][len(projections) % 4]
        priorites = utilisations + rng.random(DIM_SORTIE)
        support = np.argsort(priorites)[:poids]

        vecteur = np.zeros(DIM_SORTIE, dtype=np.int64)
        vecteur[support] = rng.integers(1, P, size=poids)

        premier = int(np.flatnonzero(vecteur)[0])
        inverse = pow(int(vecteur[premier]), P - 2, P)
        vecteur = (vecteur * inverse) % P
        cle = tuple(int(x) for x in vecteur)

        if cle not in deja:
            deja.add(cle)
            projections.append(vecteur)
            utilisations[support] += 1

    return np.asarray(projections, dtype=np.int64)

CONFIGURATIONS = construire_configurations()
PAR_ID = {cfg["id"]: cfg for cfg in CONFIGURATIONS}
PROJECTIONS = construire_projections()

LOTS = []
for longueur in LONGUEURS:
    groupe = [
        cfg for cfg in CONFIGURATIONS
        if cfg["longueur"] == longueur
    ]
    for debut in range(0, len(groupe), TAILLE_LOT):
        LOTS.append(groupe[debut:debut + TAILLE_LOT])

p0 = (1.0 - 1.0 / P) ** DIM_SORTIE
p1 = DIM_SORTIE / P * (1.0 - 1.0 / P) ** (DIM_SORTIE - 1)
PROB_ZEROS = np.asarray([p0, p1, 1.0 - p0 - p1])
PROB_UNIFORME = np.full(P, 1.0 / P)

def statistique_chi2(comptes, probabilites):
    comptes = np.asarray(comptes, dtype=np.int64)
    attendus = comptes.sum() * probabilites
    return float(np.sum((comptes - attendus) ** 2 / attendus))

def score_detection(comptes, probabilites):
    statistique = statistique_chi2(comptes, probabilites)
    ddl = len(probabilites) - 1
    return float(min(
        1.0,
        2.0 * min(
            chi2.cdf(statistique, ddl),
            chi2.sf(statistique, ddl)
        )
    ))

def p_monte_carlo(comptes, probabilites, seed):
    rng = np.random.default_rng(seed)
    n = int(np.sum(comptes))
    attendus = n * probabilites
    observe = statistique_chi2(comptes, probabilites)
    tolerance = 1e-10 * max(1.0, abs(observe))

    superieurs = 0
    inferieurs = 0

    for debut in range(0, B_MONTE_CARLO, 256):
        taille = min(256, B_MONTE_CARLO - debut)
        simulations = rng.multinomial(n, probabilites, size=taille)
        statistiques = np.sum(
            (simulations - attendus) ** 2 / attendus, axis=1
        )
        superieurs += int(np.sum(statistiques >= observe - tolerance))
        inferieurs += int(np.sum(statistiques <= observe + tolerance))

    p_sup = (superieurs + 1.0) / (B_MONTE_CARLO + 1.0)
    p_inf = (inferieurs + 1.0) / (B_MONTE_CARLO + 1.0)
    return observe, min(1.0, 2.0 * min(p_sup, p_inf))

def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p_values)):
        raise ValueError("Une p-value n'est pas finie.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("Une p-value est hors de [0, 1].")

    ordre = np.argsort(p_values)
    ajustees = np.empty(len(p_values), dtype=float)
    maximum = 0.0

    for rang, indice in enumerate(ordre):
        maximum = max(
            maximum, (len(p_values) - rang) * float(p_values[indice])
        )
        ajustees[indice] = min(1.0, maximum)
    return ajustees

def definir_test(indice):
    if indice < DIM_SORTIE:
        return {"type": "coordonnee", "indice": int(indice)}
    if indice < DIM_SORTIE + N_PROJECTIONS:
        j = indice - DIM_SORTIE
        return {
            "type": "projection",
            "indice": int(j),
            "vecteur": PROJECTIONS[j].tolist(),
        }
    return {"type": "nombre_zeros", "categories": ["0", "1", ">=2"]}

def comptes_validation(differences, test):
    if test["type"] == "coordonnee":
        valeurs = differences[:, test["indice"]]
        probabilites = PROB_UNIFORME
    elif test["type"] == "projection":
        vecteur = np.asarray(test["vecteur"], dtype=np.int64)
        valeurs = (differences.astype(np.int64) @ vecteur) % P
        probabilites = PROB_UNIFORME
    else:
        valeurs = np.minimum(np.sum(differences == 0, axis=1), 2)
        probabilites = PROB_ZEROS

    return (
        np.bincount(valeurs, minlength=len(probabilites)),
        probabilites
    )

def detecter_lot(lot):
    identifiant_lot = sha256(canonique([cfg["id"] for cfg in lot]))
    chemin = CACHE / ("detection_" + identifiant_lot + ".json")

    if chemin.exists():
        document = lire_json(chemin)
        if (
            document["campagne"] != IDENTIFIANT
            or document["lot"] != identifiant_lot
            or [r["configuration_id"] for r in document["resultats"]]
               != [cfg["id"] for cfg in lot]
        ):
            raise ValueError("Cache de détection incompatible.")
        return document["resultats"]

    rng = np.random.default_rng(
        graine(GRAINE_DETECTION, identifiant_lot)
    )
    nombre_tests = DIM_SORTIE + N_PROJECTIONS

    comptes = np.zeros((len(lot), nombre_tests, P), dtype=np.int64)
    comptes_zeros = np.zeros((len(lot), 3), dtype=np.int64)
    vecteurs_nuls = np.zeros(len(lot), dtype=np.int64)
    temoins = [None] * len(lot)

    for _ in range(N_DETECTION):
        message = rng.integers(
            0, P, size=(lot[0]["longueur"], DIM_K), dtype=np.int64
        )
        h0 = sortie(message)

        for j, cfg in enumerate(lot):
            d = difference(message, cfg, h0)
            valeurs = np.concatenate((d, (PROJECTIONS @ d) % P))
            comptes[j, np.arange(nombre_tests), valeurs] += 1

            zeros = int(np.sum(d == 0))
            comptes_zeros[j, min(zeros, 2)] += 1

            if zeros == DIM_SORTIE:
                vecteurs_nuls[j] += 1
                if temoins[j] is None:
                    temoins[j] = message.tolist()

    resultats = []

    for j, cfg in enumerate(lot):
        scores = [
            score_detection(c, PROB_UNIFORME)
            for c in comptes[j]
        ]
        scores.append(score_detection(comptes_zeros[j], PROB_ZEROS))
        indice = int(np.argmin(scores))

        resultats.append({
            "configuration_id": cfg["id"],
            "ordre": cfg["ordre"],
            "longueur": cfg["longueur"],
            "test": definir_test(indice),

            "score_selection": float(scores[indice]),
            "nombre_tests_explores": len(scores),
            "differences_vectorielles_nulles": int(vecteurs_nuls[j]),
            "premier_message_temoin": temoins[j],
        })

    ecrire_json(chemin, {
        "campagne": IDENTIFIANT,
        "lot": identifiant_lot,
        "resultats": resultats,
    })
    return resultats

def selectionner(detection):
    candidats = []
    vus = set()

    def ajouter(cfg_id, test, origine, score=None):
        definition = {"configuration_id": cfg_id, "test": test}
        identifiant = sha256(canonique(definition))

        if identifiant in vus or len(candidats) >= N_CANDIDATS:
            return

        vus.add(identifiant)
        candidats.append({
            "id": identifiant,
            **definition,
            "origine": origine,
            "score_selection": score,
        })

    for ordre in (1, 2):
        groupe = [cfg for cfg in CONFIGURATIONS if cfg["ordre"] == ordre]
        for cfg in (groupe[0], groupe[-1]):
            ajouter(
                cfg["id"], {"type": "coordonnee", "indice": 0},
                "controle_fixe"
            )

    classement = sorted(
        detection,
        key=lambda r: (r["score_selection"], r["configuration_id"])
    )

    for resultat in [r for r in classement if r["ordre"] == 2][:4]:
        ajouter(
            resultat["configuration_id"], resultat["test"],
            "selection_ordre2", resultat["score_selection"]
        )

    for resultat in classement:
        ajouter(
            resultat["configuration_id"], resultat["test"],
            "selection_globale", resultat["score_selection"]
        )

    ecrire_json(DOSSIER / "candidats_avant_validation.json", candidats)
    return candidats

def echantillon_validation(cfg):
    chemin = CACHE / ("validation_" + cfg["id"] + ".npz")

    if chemin.exists():
        try:
            with np.load(str(chemin), allow_pickle=False) as archive:
                if str(archive["campagne"].item()) != IDENTIFIANT:
                    raise ValueError("Identifiant de campagne incorrect.")
                if str(archive["configuration"].item()) != cfg["id"]:
                    raise ValueError("Configuration incorrecte.")

                donnees = archive["differences"].copy()
                temoins = json.loads(str(archive["temoins"].item()))
                checksum = str(archive["checksum"].item())

            if donnees.dtype != np.uint8:
                raise ValueError("Type incorrect.")
            if donnees.shape != (N_VALIDATION, DIM_SORTIE):
                raise ValueError("Dimensions incorrectes.")
            if np.any(donnees >= P):
                raise ValueError("Valeurs hors de F_251.")
            if hashlib.sha256(donnees.tobytes()).hexdigest() != checksum:
                raise ValueError("Empreinte des données incorrecte.")

            return donnees, temoins
        except Exception as erreur:
            raise RuntimeError(
                "Cache invalide : {}. Renommez-le pour le recalculer."
                .format(chemin)
            ) from erreur

    rng = np.random.default_rng(
        graine(GRAINE_VALIDATION, cfg["id"])
    )
    donnees = np.empty((N_VALIDATION, DIM_SORTIE), dtype=np.uint8)
    temoins = []

    for i in range(N_VALIDATION):
        message = rng.integers(
            0, P, size=(cfg["longueur"], DIM_K), dtype=np.int64
        )
        d = difference(message, cfg, sortie(message))
        donnees[i] = d

        if not np.any(d) and len(temoins) < 3:
            temoins.append(message.tolist())

        if (i + 1) % 10000 == 0:
            print(
                "  Validation {} : {}/{} messages"
                .format(cfg["id"][:10], i + 1, N_VALIDATION),
                flush=True
            )

    temporaire = chemin.with_name(chemin.name + ".tmp")
    with temporaire.open("wb") as fichier:
        np.savez_compressed(
            fichier,
            campagne=np.asarray(IDENTIFIANT),
            configuration=np.asarray(cfg["id"]),
            differences=donnees,
            temoins=np.asarray(canonique(temoins)),
            checksum=np.asarray(
                hashlib.sha256(donnees.tobytes()).hexdigest()
            ),
        )
    os.replace(str(temporaire), str(chemin))
    return donnees, temoins

def executer():
    appels_detection = N_DETECTION * (
        len(LOTS)
        + sum(1 if cfg["ordre"] == 1 else 3 for cfg in CONFIGURATIONS)
    )

    protocole = {
        "campagne": IDENTIFIANT,
        "hachage_source_sha256": HASH_SOURCE_SHA256,
        "test_source_sha256": TEST_SOURCE_SHA256,
        "p": P,
        "alpha_couche_lineaire": 4,
        "tours": 8,
        "longueurs_en_elements_de_K": LONGUEURS,
        "deltas": DELTAS,
        "n_detection": N_DETECTION,
        "n_validation": N_VALIDATION,
        "nombre_configurations": len(CONFIGURATIONS),
        "nombre_tests_par_configuration": DIM_SORTIE + N_PROJECTIONS + 1,
        "maximum_candidats": N_CANDIDATS,
        "simulations_monte_carlo": B_MONTE_CARLO,
        "seuil_global_holm": SEUIL_GLOBAL,
        "appels_hachage_detection_sans_cache": appels_detection,
        "numpy_version": np.__version__,
        "interpretation": (
            "Uniformité de statistiques différentielles sélectionnées "
            "sur des messages aléatoires de longueur fixée."
        ),
    }

    ecrire_json(DOSSIER / "protocole.json", protocole)
    ecrire_json(DOSSIER / "configurations.json", CONFIGURATIONS)
    ecrire_json(DOSSIER / "projections.json", PROJECTIONS.tolist())

    (DOSSIER / "hachage_source.py").write_text(
        RAINHASH_SOURCE, encoding="utf-8"
    )
    (DOSSIER / "test_source.py").write_text(
        TEST_DIFFERENTIEL_SOURCE, encoding="utf-8"
    )

    print("Hachage : sponge, F_251, alpha=4, huit tours.")
    print("Configurations :", len(CONFIGURATIONS))
    print("Appels au hachage en détection :", appels_detection)
    print("Dossier :", DOSSIER)
    print("Début de la détection.", flush=True)

    debut = perf_counter()
    detection = []

    for numero, lot in enumerate(LOTS, 1):
        detection.extend(detecter_lot(lot))
        print(
            "Lot {}/{} terminé ; durée écoulée : {:.1f} min."
            .format(numero, len(LOTS), (perf_counter() - debut) / 60),
            flush=True
        )

    ecrire_csv(DOSSIER / "detection_exploratoire.csv", detection)
    candidats = selectionner(detection)

    if 2.0 / (B_MONTE_CARLO + 1) > SEUIL_GLOBAL / len(candidats):
        raise ValueError(
            "Résolution Monte-Carlo insuffisante pour la correction de Holm."
        )

    resultats = []

    ids = list(dict.fromkeys(c["configuration_id"] for c in candidats))

    for numero, cfg_id in enumerate(ids, 1):
        cfg = PAR_ID[cfg_id]
        print(
            "Configuration de validation {}/{}."
            .format(numero, len(ids)),
            flush=True
        )
        donnees, temoins = echantillon_validation(cfg)
        nombre_nuls = int(np.sum(np.all(donnees == 0, axis=1)))

        for candidat in candidats:
            if candidat["configuration_id"] != cfg_id:
                continue

            comptes, probabilites = comptes_validation(
                donnees, candidat["test"]
            )
            statistique, p_value = p_monte_carlo(
                comptes, probabilites,
                graine(GRAINE_MONTE_CARLO, candidat["id"])
            )
            frequences = comptes / float(N_VALIDATION)

            resultats.append({
                "candidat_id": candidat["id"],
                "configuration_id": cfg_id,
                "ordre": cfg["ordre"],
                "longueur": cfg["longueur"],
                "delta1": cfg["delta1"],
                "delta2": cfg["delta2"],
                "test": candidat["test"],
                "origine": candidat["origine"],
                "n": N_VALIDATION,
                "chi2": statistique,
                "p_monte_carlo_bilaterale": p_value,
                "variation_totale_empirique": float(
                    0.5 * np.sum(np.abs(frequences - probabilites))
                ),
                "ecart_frequence_maximal": float(
                    np.max(np.abs(frequences - probabilites))
                ),
                "differences_vectorielles_nulles": nombre_nuls,
                "nature_vecteur_nul": (
                    "collision H(M+delta)=H(M)"
                    if cfg["ordre"] == 1 else
                    "relation du second ordre, pas necessairement collision"
                ),
                "messages_temoins": temoins,
            })

        del donnees

    ajustees = holm([
        resultat["p_monte_carlo_bilaterale"]
        for resultat in resultats
    ])

    for resultat, p_ajustee in zip(resultats, ajustees):
        resultat["p_holm"] = float(p_ajustee)
        resultat["rejet_holm"] = bool(p_ajustee <= SEUIL_GLOBAL)

    ecrire_csv(DOSSIER / "validation_confirmatoire.csv", resultats)
    ecrire_json(DOSSIER / "validation_confirmatoire.json", resultats)

    rejets = sum(int(r["rejet_holm"]) for r in resultats)

    collisions_detection = [
        r for r in detection
        if r["ordre"] == 1 and r["differences_vectorielles_nulles"] > 0
    ]
    collisions_validation = {
        r["configuration_id"]
        for r in resultats
        if r["ordre"] == 1 and r["differences_vectorielles_nulles"] > 0
    }

    if rejets:
        conclusion = (
            "Des écarts au modèle nul sont détectés après correction "
            "de Holm parmi les tests validés."
        )
    else:
        conclusion = (
            "Aucun écart significatif au modèle nul n'est détecté "
            "parmi les tests sélectionnés et validés."
        )

    resume = {
        "configurations_explorees": len(CONFIGURATIONS),
        "tests_confirmatoires": len(resultats),
        "rejets_holm": rejets,
        "configurations_avec_collision_detection": len(collisions_detection),
        "configurations_avec_collision_validation": len(collisions_validation),
        "duree_secondes": perf_counter() - debut,
        "conclusion": conclusion,
        "modele_nul": (
            "Pour chaque configuration, les différences observées sont "
            "comparées à des vecteurs indépendants uniformes de F_251^32. "
            "Il s'agit d'un modèle statistique de référence."
        ),
        "limite_monte_carlo": (
            "Les p-values sont discrètes ; leur valeur minimale "
            "bilatérale est 2/(B+1)."
        ),
    }
    ecrire_json(DOSSIER / "resume.json", resume)

    print("\n" + conclusion)
    print("Tests confirmatoires :", len(resultats))
    print("Rejets après Holm :", rejets)

    if collisions_detection or collisions_validation:
        print(
            "Des différences nulles du premier ordre ont été observées : "
            "examiner les messages témoins enregistrés."
        )

    print("Résultats enregistrés dans :", DOSSIER)
    return resume

RESUME_DIFFERENTIEL = executer()
'''

_espace_differentiel = {
    "__name__": "rainhash_test_differentiel",
    "RAINHASH_MODULE": RAINHASH_MODULE,
    "RAINHASH_SOURCE": RAINHASH_SOURCE,
    "TEST_DIFFERENTIEL_SOURCE": TEST_DIFFERENTIEL_SOURCE,
}

exec(
    compile(TEST_DIFFERENTIEL_SOURCE, "<test_differentiel_alpha4>", "exec"),
    _espace_differentiel
)

RESUME_DIFFERENTIEL = _espace_differentiel["RESUME_DIFFERENTIEL"]
