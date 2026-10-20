import types

RAINHASH_SOURCE = r'''from sage.all import GF, Integer, RealField, floor
import random

p = Integer(251)
T0 = GF(p)
T = 8
ALPHA_L = T0(4)
NB_TOURS = 8
NB_COORDONNEES_T3 = 8
NB_CHIFFRES_PI = NB_TOURS * T * NB_COORDONNEES_T3
TAUX_SPONGE = 4
CAPACITE_SPONGE = 4
TAILLE_EMPREINTE_SPONGE = 4
assert p.is_prime() and p % 8 == 3
assert ALPHA_L != 0 and ALPHA_L**4 != T0(1)
DELTA2 = (T0(1), T0(1))
ZERO_T1 = (T0(0), T0(0))
UN_T1 = (T0(1), T0(0))
DELTA3 = (ZERO_T1, UN_T1)

def coordonnees_vers_tour(v):

    if len(v) != 8:
        raise ValueError("Il faut exactement 8 coordonnées")

    x = [T0(c) for c in v]

    return (
        (
            (x[0], x[1]),
            (x[2], x[3])
        ),
        (
            (x[4], x[5]),
            (x[6], x[7])
        )
    )

def tour_vers_coordonnees(X):

    A0, A1 = X

    a0, a1 = A0
    b0, b1 = A1

    return [
        a0[0], a0[1],
        a1[0], a1[1],
        b0[0], b0[1],
        b1[0], b1[1]
    ]

def addition_T1(A, B):
    return (
        A[0] + B[0],
        A[1] + B[1]
    )

def soustraction_T1(A, B):
    return (
        A[0] - B[0],
        A[1] - B[1]
    )

def multiplication_T1(A, B):

    a0, a1 = A
    b0, b1 = B

    v0 = a0 * b0
    v1 = a1 * b1

    c0 = v0 - v1

    c1 = (a0 + a1) * (b0 + b1) - v0 - v1

    return c0, c1

def carre_T1(A):

    a0, a1 = A

    return (
        a0**2 - a1**2,
        T0(2) * a0 * a1
    )

def addition_T2(A, B):
    return (
        addition_T1(A[0], B[0]),
        addition_T1(A[1], B[1])
    )

def soustraction_T2(A, B):
    return (
        soustraction_T1(A[0], B[0]),
        soustraction_T1(A[1], B[1])
    )

def multiplication_T2(A, B):

    a0, a1 = A
    b0, b1 = B

    v0 = multiplication_T1(a0, b0)
    v1 = multiplication_T1(a1, b1)

    c0 = addition_T1(
        v0,
        multiplication_T1(DELTA2, v1)
    )

    produit_sommes = multiplication_T1(
        addition_T1(a0, a1),
        addition_T1(b0, b1)
    )

    c1 = soustraction_T1(
        soustraction_T1(produit_sommes, v0),
        v1
    )

    return c0, c1

def carre_T2(A):

    a0, a1 = A

    a0_carre = carre_T1(a0)
    a1_carre = carre_T1(a1)

    c0 = addition_T1(
        a0_carre,
        multiplication_T1(DELTA2, a1_carre)
    )

    produit = multiplication_T1(a0, a1)
    c1 = addition_T1(produit, produit)

    return c0, c1

def addition_tour_T3(A, B):
    return (
        addition_T2(A[0], B[0]),
        addition_T2(A[1], B[1])
    )

def soustraction_tour_T3(A, B):
    return (
        soustraction_T2(A[0], B[0]),
        soustraction_T2(A[1], B[1])
    )

def multiplication_tour_T3(A, B):

    a0, a1 = A
    b0, b1 = B

    v0 = multiplication_T2(a0, b0)
    v1 = multiplication_T2(a1, b1)

    c0 = addition_T2(
        v0,
        multiplication_T2(DELTA3, v1)
    )

    produit_sommes = multiplication_T2(
        addition_T2(a0, a1),
        addition_T2(b0, b1)
    )

    c1 = soustraction_T2(
        soustraction_T2(produit_sommes, v0),
        v1
    )

    return c0, c1

def carre_tour_T3(A):

    a0, a1 = A

    a0_carre = carre_T2(a0)
    a1_carre = carre_T2(a1)

    c0 = addition_T2(
        a0_carre,
        multiplication_T2(DELTA3, a1_carre)
    )

    produit = multiplication_T2(a0, a1)
    c1 = addition_T2(produit, produit)

    return c0, c1

def addition_T3(x, y):
    X = coordonnees_vers_tour(x)
    Y = coordonnees_vers_tour(y)

    return tour_vers_coordonnees(
        addition_tour_T3(X, Y)
    )

def multiplication_T3(x, y):
    X = coordonnees_vers_tour(x)
    Y = coordonnees_vers_tour(y)

    return tour_vers_coordonnees(
        multiplication_tour_T3(X, Y)
    )

def carre_T3(x):
    X = coordonnees_vers_tour(x)

    return tour_vers_coordonnees(
        carre_tour_T3(X)
    )

def est_nul_T1(A):
    return A[0] == 0 and A[1] == 0

def est_nul_T2(A):
    return est_nul_T1(A[0]) and est_nul_T1(A[1])

def est_nul_tour_T3(A):
    return est_nul_T2(A[0]) and est_nul_T2(A[1])

def oppose_T1(A):
    return (-A[0], -A[1])

def oppose_T2(A):
    return (
        oppose_T1(A[0]),
        oppose_T1(A[1])
    )

def inverse_T0(a):

    a = T0(a)

    if a == 0:
        raise ZeroDivisionError("Zéro n'est pas inversible dans T0")

    resultat = a ** (p - 2)

    assert a * resultat == T0(1)

    return resultat

def inverse_tour_T1(A):

    if est_nul_T1(A):
        raise ZeroDivisionError("Zéro n'est pas inversible dans T1")

    a0, a1 = A

    delta1 = a0**2 + a1**2

    delta1_inverse = inverse_T0(delta1)

    return (
        a0 * delta1_inverse,
        -a1 * delta1_inverse
    )

def inverse_tour_T2(A):

    if est_nul_T2(A):
        raise ZeroDivisionError("Zéro n'est pas inversible dans T2")

    a0, a1 = A

    a0_carre = carre_T1(a0)
    a1_carre = carre_T1(a1)

    terme = multiplication_T1(DELTA2, a1_carre)

    delta2 = soustraction_T1(a0_carre, terme)

    delta2_inverse = inverse_tour_T1(delta2)

    c0 = multiplication_T1(a0, delta2_inverse)

    c1 = multiplication_T1(
        oppose_T1(a1),
        delta2_inverse
    )

    return c0, c1

def inverse_tour_T3(A):

    if est_nul_tour_T3(A):
        raise ZeroDivisionError("Zéro n'est pas inversible dans T3")

    a0, a1 = A

    a0_carre = carre_T2(a0)
    a1_carre = carre_T2(a1)

    terme = multiplication_T2(DELTA3, a1_carre)

    delta3 = soustraction_T2(a0_carre, terme)

    delta3_inverse = inverse_tour_T2(delta3)

    c0 = multiplication_T2(a0, delta3_inverse)

    c1 = multiplication_T2(
        oppose_T2(a1),
        delta3_inverse
    )

    return c0, c1

def inverse_T3(x):

    X = coordonnees_vers_tour(x)
    X_inverse = inverse_tour_T3(X)

    return tour_vers_coordonnees(X_inverse)

def normaliser_coordonnees_T3(x):

    if len(x) != 8:
        raise ValueError("Il faut exactement 8 coordonnées")

    return [T0(c) for c in x]

def est_zero_coordonnees_T3(x):

    x = normaliser_coordonnees_T3(x)

    return all(coefficient == 0 for coefficient in x)

def S_T3(x):

    x = normaliser_coordonnees_T3(x)

    if est_zero_coordonnees_T3(x):
        return [T0(0) for _ in range(8)]

    return inverse_T3(x)

def normaliser_etat_T3(etat):

    if len(etat) != T:
        raise ValueError(
            "L'état doit contenir exactement 8 colonnes"
        )

    return [
        normaliser_coordonnees_T3(colonne)
        for colonne in etat
    ]

def colonnes_vers_lignes(etat):

    etat = normaliser_etat_T3(etat)

    return [
        [etat[j][i] for j in range(T)]
        for i in range(8)
    ]

def lignes_vers_colonnes(lignes):

    if len(lignes) != 8:
        raise ValueError("Il faut exactement 8 lignes")

    if any(len(ligne) != T for ligne in lignes):
        raise ValueError(
            "Chaque ligne doit contenir exactement 8 éléments"
        )

    return [
        [T0(lignes[i][j]) for i in range(8)]
        for j in range(T)
    ]

def rotation_droite(x, decalage):

    decalage %= len(x)

    if decalage == 0:
        return list(x)

    return x[-decalage:] + x[:-decalage]

def rotation_gauche(x, decalage):

    decalage %= len(x)

    if decalage == 0:
        return list(x)

    return x[decalage:] + x[:decalage]

def L_sans_MDS(etat):

    lignes = colonnes_vers_lignes(etat)
    return lignes_vers_colonnes([
        [a + ALPHA_L*b for a, b in zip(
            rotation_droite(lignes[i], i),
            rotation_gauche(lignes[i], i))]
        for i in range(8)
    ])

def L_inverse_sans_MDS(etat):

    lignes = colonnes_vers_lignes(etat)
    facteur = inverse_T0(T0(1) - ALPHA_L**4)
    sortie = []
    for i in range(8):
        termes = [rotation_gauche(lignes[i], (2*k+1)*i)
                  for k in range(4)]
        sortie.append([
            facteur * sum(((-ALPHA_L)**k * termes[k][j]
                           for k in range(4)), T0(0))
            for j in range(8)
        ])
    return lignes_vers_colonnes(sortie)

def couche_S_etat(etat):

    etat = normaliser_etat_T3(etat)

    return [
        S_T3(colonne)
        for colonne in etat
    ]

def addition_etats_T3(X, Y):

    X = normaliser_etat_T3(X)
    Y = normaliser_etat_T3(Y)

    return [
        addition_T3(X[j], Y[j])
        for j in range(T)
    ]

def soustraction_etats_T3(X, Y):

    X = normaliser_etat_T3(X)
    Y = normaliser_etat_T3(Y)

    return [
        [
            X[colonne][ligne] - Y[colonne][ligne]
            for ligne in range(8)
        ]
        for colonne in range(T)
    ]

def tour_direct_T3(etat, constante_tour):

    etat = normaliser_etat_T3(etat)
    constante_tour = normaliser_etat_T3(constante_tour)

    apres_S = couche_S_etat(etat)

    apres_L = L_sans_MDS(apres_S)

    return addition_etats_T3(
        apres_L,
        constante_tour
    )

def tour_inverse_T3(etat, constante_tour):

    etat = normaliser_etat_T3(etat)
    constante_tour = normaliser_etat_T3(constante_tour)

    sans_constante = soustraction_etats_T3(
        etat,
        constante_tour
    )

    avant_L = L_inverse_sans_MDS(
        sans_constante
    )

    return couche_S_etat(avant_L)

def chiffres_pi_base_p(nombre_chiffres, marge_bits=256):

    base = Integer(p)

    precision_bits = (
        nombre_chiffres * base.nbits()
        + marge_bits
    )

    RR = RealField(precision_bits)

    valeur = (
        (RR.pi() - RR(3))
        * (RR(base) ** nombre_chiffres)
    )

    entier = Integer(floor(valeur))

    chiffres = [Integer(0)] * nombre_chiffres

    for position in range(nombre_chiffres - 1, -1, -1):

        entier, chiffre = divmod(entier, base)

        chiffres[position] = Integer(chiffre)

    assert entier == 0

    return chiffres

def construire_constantes_tours(chiffres):

    constantes = []
    position = 0

    for numero_tour in range(NB_TOURS):

        constante = []

        for colonne in range(T):

            element_T3_constante = [
                T0(chiffres[position + ligne])
                for ligne in range(NB_COORDONNEES_T3)
            ]

            constante.append(element_T3_constante)

            position += NB_COORDONNEES_T3

        constantes.append(constante)

    assert position == len(chiffres)

    return constantes

CHIFFRES_PI = chiffres_pi_base_p(NB_CHIFFRES_PI, marge_bits=256)
if CHIFFRES_PI != chiffres_pi_base_p(NB_CHIFFRES_PI, marge_bits=512):
    raise ArithmeticError("Chiffres de pi instables : augmenter la précision")
CONSTANTES_TOURS = construire_constantes_tours(CHIFFRES_PI)

def permutation_complete_T3(
    etat,
    constantes=CONSTANTES_TOURS
):

    if len(constantes) != NB_TOURS:
        raise ValueError(
            "Il faut exactement {} constantes."
            .format(NB_TOURS)
        )

    etat = normaliser_etat_T3(etat)

    etat = L_sans_MDS(etat)

    for numero_tour in range(NB_TOURS):

        etat = tour_direct_T3(
            etat,
            constantes[numero_tour]
        )

    return etat

def permutation_inverse_T3(
    etat,
    constantes=CONSTANTES_TOURS
):

    if len(constantes) != NB_TOURS:
        raise ValueError(
            "Il faut exactement {} constantes."
            .format(NB_TOURS)
        )

    etat = normaliser_etat_T3(etat)

    for numero_tour in range(
        NB_TOURS - 1,
        -1,
        -1
    ):

        etat = tour_inverse_T3(
            etat,
            constantes[numero_tour]
        )

    return L_inverse_sans_MDS(etat)

def normaliser_bloc_T3(bloc, longueur, nom="bloc"):

    if len(bloc) != longueur:
        raise ValueError(
            "{} doit contenir exactement {} éléments de T3."
            .format(nom, longueur)
        )

    return [
        normaliser_coordonnees_T3(element)
        for element in bloc
    ]

def scalaire_vers_T3(a):

    return [
        T0(a)
    ] + [
        T0(0)
        for _ in range(7)
    ]

def zero_T3():

    return scalaire_vers_T3(0)

def un_T3():

    return scalaire_vers_T3(1)

def normaliser_message_T3(message):

    return [
        normaliser_coordonnees_T3(element)
        for element in message
    ]

def padding_sponge_pi_T3(message):

    message = normaliser_message_T3(message)

    longueur = len(message)

    if longueur == 0:

        message_complete = (
            [un_T3()]
            +
            [
                zero_T3()
                for _ in range(TAUX_SPONGE - 1)
            ]
        )

        mu = 1

    elif longueur % TAUX_SPONGE == 0:

        message_complete = [
            list(element)
            for element in message
        ]

        mu = 0

    else:

        nombre_zeros = (
            TAUX_SPONGE
            - (longueur % TAUX_SPONGE)
            - 1
        )

        message_complete = (
            [
                list(element)
                for element in message
            ]
            +
            [un_T3()]
            +
            [
                zero_T3()
                for _ in range(nombre_zeros)
            ]
        )

        mu = 1

    assert len(message_complete) > 0

    assert (
        len(message_complete)
        % TAUX_SPONGE
        == 0
    )

    blocs = [
        message_complete[
            position:
            position + TAUX_SPONGE
        ]
        for position in range(
            0,
            len(message_complete),
            TAUX_SPONGE
        )
    ]

    assert all(
        len(bloc) == TAUX_SPONGE
        for bloc in blocs
    )

    return blocs, mu

def etat_initial_sponge_T3():

    etat = [
        zero_T3()
        for _ in range(
            TAUX_SPONGE
            + CAPACITE_SPONGE
            - 1
        )
    ]

    etat.append(
        scalaire_vers_T3(
            TAILLE_EMPREINTE_SPONGE
        )
    )

    assert len(etat) == T

    return etat

def injection_bloc_sponge_T3(
    bloc,
    indicateur_final=None
):

    bloc = normaliser_bloc_T3(
        bloc,
        TAUX_SPONGE,
        "bloc_sponge"
    )

    injection = (
        [
            list(element)
            for element in bloc
        ]
        +
        [
            zero_T3()
            for _ in range(CAPACITE_SPONGE)
        ]
    )

    if indicateur_final is not None:

        if indicateur_final not in (0, 1):
            raise ValueError(
                "L'indicateur μ doit être 0 ou 1."
            )

        injection[-1] = scalaire_vers_T3(
            indicateur_final
        )

    assert len(injection) == T

    return injection

def absorption_sponge_pi_T3(message):

    blocs, mu = padding_sponge_pi_T3(
        message
    )

    etat = etat_initial_sponge_T3()

    for numero_bloc, bloc in enumerate(blocs):

        dernier_bloc = (
            numero_bloc
            == len(blocs) - 1
        )

        if dernier_bloc:

            injection = injection_bloc_sponge_T3(
                bloc,
                indicateur_final=mu
            )

        else:

            injection = injection_bloc_sponge_T3(
                bloc
            )

        etat = addition_etats_T3(
            etat,
            injection
        )

        etat = permutation_complete_T3(
            etat
        )

    return etat

def hachage_sponge_pi_T3(message):

    etat = absorption_sponge_pi_T3(
        message
    )

    etat = permutation_complete_T3(
        etat
    )

    empreinte = [
        list(element)
        for element in etat[
            :TAILLE_EMPREINTE_SPONGE
        ]
    ]

    assert len(empreinte) == 4

    return empreinte

def coordonnees_empreinte_F251(empreinte):

    empreinte = normaliser_bloc_T3(
        empreinte,
        TAILLE_EMPREINTE_SPONGE,
        "empreinte"
    )

    return [
        coefficient
        for element in empreinte
        for coefficient in element
    ]
'''

RAINHASH_MODULE = types.ModuleType('rainhash_alpha4_notebook')
exec(compile(RAINHASH_SOURCE, '<rainhash_alpha4>', 'exec'), RAINHASH_MODULE.__dict__)

hachage_sponge_pi_T3 = RAINHASH_MODULE.hachage_sponge_pi_T3
coordonnees_empreinte_F251 = RAINHASH_MODULE.coordonnees_empreinte_F251
print('Hachage chargé : p=251, alpha=4, huit tours, sponge.')
