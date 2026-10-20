from rainhash_sponge_fp251_alpha4_8tours import RAINHASH_MODULE, RAINHASH_SOURCE

DUREE_SESSION_SECONDES = 300
REPETITIONS_AVALANCHE = 1

TEST_AVALANCHE_SOURCE = r'''import csv, hashlib, json, math, os, threading
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.stats import binom

P = 251
INTERVALLE_AFFICHAGE = 10.0
PROB_CHANGEMENT = 250.0/251
MOYENNE_IDEALE = 32*PROB_CHANGEMENT

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
            self.afficher('FIN DE SESSION')
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

def hacher(message):
    entrees = [[RAINHASH_MODULE.T0(int(v)) for v in ligne] for ligne in message]
    return convertir_sortie(RAINHASH_MODULE.hachage_sponge_pi_T3(entrees))

def ajouter_test(lignes, famille, detail, inchanges, comparaisons):

    k, n = int(inchanges), int(comparaisons)
    p = min(1.0, 2.0*min(float(binom.cdf(k,n,1.0/P)),
                         float(binom.sf(k-1,n,1.0/P))))
    lignes.append({'famille': famille, 'detail': detail,
                   'coordonnees_inchangees': k, 'comparaisons': n,
                   'p_bilaterale': p, 'methode': 'binomiale exacte, queues doublees'})

def statistiques(distances):
    return {'essais':len(distances),'distance_moyenne':float(np.mean(distances)),
        'taux_changement_pourcentage':100*float(np.mean(distances))/32,
        'ecart_type_distance':float(np.std(distances,ddof=1)),
        'distance_min':int(np.min(distances)), 'distance_max':int(np.max(distances)),
        'distances_nulles':int(np.sum(distances==0)),
        'distances_32':int(np.sum(distances==32))}

def sauvegarder(chemin, masques, collisions, identifiant):
    temporaire = chemin.with_name(chemin.name + '.tmp')
    texte = canonique(collisions)
    with temporaire.open('wb') as f:
        np.savez_compressed(f, masques=masques, collisions=np.asarray(texte),
            identifiant=np.asarray(identifiant),
            sha256=np.asarray(hashlib.sha256(masques.tobytes()+texte.encode('utf-8')).hexdigest()))
    os.replace(str(temporaire), str(chemin))

def executer():
    debut = perf_counter()
    if DUREE_SESSION_SECONDES <= 0 or REPETITIONS < 1:
        raise ValueError('Durée et répétitions doivent être positives.')

    longueurs = list(range(1, 17))
    deltas = list(range(1, 251))
    positions = [(l,e,c) for l in longueurs for e in range(l) for c in range(8)]
    plan = [(l,e,c,d,r) for l,e,c in positions
            for r in range(REPETITIONS) for d in deltas]
    n = len(plan)
    print('RainHash : alpha=4, 8 tours, sponge r=c=4.', flush=True)
    print('{} essais, {} hachages (+ 5 témoins par session).'.format(n,2*n), flush=True)
    print('Limite souple : {} s ; arrêt après la paire en cours.'.format(DUREE_SESSION_SECONDES), flush=True)
    with Progression('vérification du module existant',1) as suivi:
        temoins = verifier_hachage()
        suivi.avancer(1)
    identite = {'version_test': 'complet-272000-v1', 'source_hachage': empreinte(RAINHASH_SOURCE),
        'source_test': empreinte(TEST_AVALANCHE_SOURCE), 'temoins': temoins,
        'longueurs': longueurs, 'deltas': deltas, 'repetitions': REPETITIONS,
        'graine': 2026093021, 'numpy': np.__version__, 'scipy': scipy.__version__}
    identifiant = empreinte(canonique(identite))
    dossier = Path('resultats_avalanche_alpha4_272000')/identifiant
    dossier.mkdir(parents=True,exist_ok=True)
    chemin = dossier/'reprise.npz'
    ecrire_json(dossier/'protocole.json', dict(identite, essais=n,
        modele='sorties indépendantes uniformes dans F_251^32',
        nature='272000 essais ; distances symboliques dans F_251^32'))
    masques = np.full((n,32),255,dtype=np.uint8)
    collisions = []
    if chemin.exists():
        with np.load(str(chemin),allow_pickle=False) as z:
            m = z['masques'].copy()
            texte = str(z['collisions'].item())
            if (str(z['identifiant'].item()) != identifiant or m.shape != (n,32)
                    or m.dtype != np.uint8):
                raise ValueError('Cache incompatible ; ne pas réutiliser ces données.')
            if hashlib.sha256(m.tobytes()+texte.encode('utf-8')).hexdigest() != str(z['sha256'].item()):
                raise ValueError('Cache corrompu.')
            if not np.all(np.all(m == 255,axis=1) | np.all(m <= 1,axis=1)):
                raise ValueError('Cache : lignes invalides.')
            masques, collisions = m, json.loads(texte)
    manquants = np.flatnonzero(masques[:,0] == 255)
    print('{} / {} essais déjà sauvegardés.'.format(n-len(manquants),n),flush=True)
    print('Relancer cette cellule reprend les calculs. Ne pas lancer deux copies simultanément.',flush=True)
    faits = 0
    dernier = perf_counter()
    try:
        with Progression('hachages de cette session',2*len(manquants),'hachages') as suivi:
            for index in manquants:
                if perf_counter()-debut >= DUREE_SESSION_SECONDES:
                    break
                i = int(index)
                l,e,c,d,r = plan[i]
                suivi.titre = 'L={}, élément {}, coefficient {}'.format(l,e+1,c+1)

                rng = np.random.default_rng(graine(2026093021,l,e,c,d,r))
                message = rng.integers(0,P,size=(l,8),dtype=np.int64)
                modifie = message.copy()
                modifie[e,c] = (int(modifie[e,c])+d)%P
                h0 = hacher(message)
                suivi.avancer(2*faits+1)
                h1 = hacher(modifie)

                masque = (h0 != h1).astype(np.uint8)
                if not np.any(masque):
                    collisions.append({'essai': i,'message':message.tolist(),
                        'message_modifie':modifie.tolist(),'empreinte':h0.tolist()})
                masques[i] = masque
                faits += 1
                suivi.avancer(2*faits)
                if perf_counter()-dernier >= 10:
                    sauvegarder(chemin,masques,collisions,identifiant)
                    dernier = perf_counter()

            suivi.total = 2*faits
    except KeyboardInterrupt:
        print('Interruption demandée : sauvegarde des essais terminés.',flush=True)
    finally:
        sauvegarder(chemin,masques,collisions,identifiant)
    termines = int(np.sum(masques[:,0] != 255))
    if termines != n:
        resume = {'termine':False,'essais_termines':termines,'essais_prevus':n,
                  'dossier':str(dossier)}
        ecrire_json(dossier/'avancement.json',resume)
        print('PAUSE : {}/{} essais. Relancez la même cellule pour continuer.'.format(termines,n),flush=True)
        print('Aucun test de significativité avant la fin du plan fixé.',flush=True)
        return resume

    distances = masques.sum(axis=1,dtype=np.int64)
    tests = []
    ajouter_test(tests,'global','toutes coordonnées',32*n-int(distances.sum()),32*n)
    par_longueur = []
    for l in longueurs:
        indices = [i for i,p in enumerate(plan) if p[0] == l]
        dist = distances[indices]
        par_longueur.append(dict(statistiques(dist),longueur=l))
        ajouter_test(tests,'longueur',str(l),32*len(dist)-int(dist.sum()),32*len(dist))
    sorties = []
    for j in range(32):
        changes = int(masques[:,j].sum())
        ajouter_test(tests,'sortie',str(j+1),n-changes,n)
        sorties.append({'coordonnee':j+1,'essais':n,'taux_pourcentage':100.0*changes/n})

    for ligne,p in zip(tests,correction_holm([t['p_bilaterale'] for t in tests])):
        ligne['p_holm'] = float(p)
        ligne['rejet_holm'] = bool(p <= 0.05)
    hist = np.bincount(distances,minlength=33)
    ecrire_csv(dossier/'tests_holm.csv',tests)
    ecrire_csv(dossier/'par_longueur.csv',par_longueur)
    ecrire_csv(dossier/'par_sortie.csv',sorties)
    ecrire_csv(dossier/'distribution_descriptive.csv',[
        {'distance':i,'observe':int(hist[i]),
         'attendu':float(n*binom.pmf(i,32,PROB_CHANGEMENT))} for i in range(33)])
    ecrire_json(dossier/'collisions.json',collisions)
    resume = statistiques(distances)
    resume.update({'termine':True,'alpha':4,'tours':8,
        'distance_ideale':MOYENNE_IDEALE,'taux_ideal_pourcentage':100*PROB_CHANGEMENT,
        'nombre_tests':len(tests),'rejets_holm':sum(t['rejet_holm'] for t in tests),
        'dossier':str(dossier)})
    ecrire_json(dossier/'resume.json',resume)
    print('TERMINÉ : {} essais.'.format(n))
    print('Distance moyenne : {:.8f}/32 ; idéal {:.8f}/32.'.format(float(distances.mean()),MOYENNE_IDEALE))
    print('Taux : {:.8f} % ; idéal {:.8f} %.'.format(100*float(distances.mean())/32,100*PROB_CHANGEMENT))
    print('Rejets après Holm : {} / {}.'.format(resume['rejets_holm'],len(tests)))
    print('Résultats :',dossier)
    return resume

RESUME_AVALANCHE = executer()
'''

if 'RAINHASH_MODULE' not in globals() or 'RAINHASH_SOURCE' not in globals():
    raise NameError('Exécutez d’abord votre cellule 1 de hachage alpha=4.')
_espace_avalanche = {
    '__name__':'test_avalanche_alpha4_272000',
    'RAINHASH_MODULE':RAINHASH_MODULE,
    'RAINHASH_SOURCE':RAINHASH_SOURCE,
    'TEST_AVALANCHE_SOURCE':TEST_AVALANCHE_SOURCE,
    'DUREE_SESSION_SECONDES':float(DUREE_SESSION_SECONDES),
    'REPETITIONS':int(REPETITIONS_AVALANCHE),
}
exec(compile(TEST_AVALANCHE_SOURCE,'<avalanche_alpha4_272000>','exec'),_espace_avalanche)
RESUME_AVALANCHE = _espace_avalanche['RESUME_AVALANCHE']
