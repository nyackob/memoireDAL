"""
Script contenant les critère d'arrêt
""" 


import numpy as np

def budget_predefini(k,N):
    """
    Vérifie si une valeur donnée k est inférieure ou égale à un budget prédefini N.
    
    Arguments:
    - k (int): La valeur à vérifier.
    - N (int): Le budget prédefini.
    
    Retour:
    - bool: True si k est inférieur ou égal à N, sinon False.
    """
    return k <= N

def incertitude_maximale(prediction_clas, seuil, affiche=0):
    """
    Calcule l'incertitude maximale parmi les prédictions et vérifie si elle est en dessous du seuil.

    Arguments:
    - prediction_clas (np.array): Les prédictions du modèle sous forme de probabilités.
    - seuil (float): Le seuil d'incertitude à ne pas dépasser.
    - affiche (int, optionnel): Si non nul, affiche l'incertitude maximale calculée.

    Retour:
    - bool: True si l'incertitude maximale est en dessous du seuil, sinon False.
    """
    # Calculer l'incertitude pour chaque point (par exemple, 1 - probabilité de la classe prédite)
    incertitudes = 1 - np.max(prediction_clas, axis=1)
    # Trouver l'incertitude maximale
    incertitude_max = np.max(incertitudes)
    # Vérifier si cette incertitude maximale est en dessous du seuil
    if affiche:
        print(incertitude_max)
    return incertitude_max < seuil

def incertitude_globale(prediction_clas, seuil, affiche=0):
    """
    Calcule l'incertitude moyenne parmi les prédictions et vérifie si elle est en dessous du seuil.

    Arguments:
    - prediction_clas (np.array): Les prédictions du modèle sous forme de probabilités.
    - seuil (float): Le seuil d'incertitude à ne pas dépasser.
    - affiche (int, optionnel): Si non nul, affiche l'incertitude moyenne calculée.

    Retour:
    - bool: True si l'incertitude moyenne est en dessous du seuil, sinon False.
    """
    # Calculer l'incertitude pour chaque point (par exemple, 1 - probabilité de la classe prédite)
    incertitudes = 1 - np.max(prediction_clas, axis=1)
    # Calculer l'incertitude moyenne
    incertitude_moyenne = np.mean(incertitudes)
    # Vérifier si cette incertitude moyenne est en dessous du seuil
    if affiche:
        print(incertitude_moyenne)
    return incertitude_moyenne < seuil

def precision_selectionnee(prediction, y_true, seuil, affiche=0):
    """
    Calcule la précision des prédictions et vérifie si elle dépasse un seuil donné.

    Arguments:
    - prediction (np.array): Les prédictions du modèle sous forme de probabilités.
    - y_true (np.array): Les étiquettes réelles.
    - seuil (float): Le seuil de précision à atteindre.
    - affiche (int, optionnel): Si non nul, affiche la précision calculée.

    Retour:
    - bool: True si la précision dépasse le seuil, sinon False.
    """
    # Transformer les probabilités en étiquettes de classe
    prediction_classes = np.argmax(prediction, axis=1)
    # Calculer la précision du modèle
    precision = np.mean(prediction_classes == y_true)
    # Vérifier si cette précision dépasse le seuil
    if affiche:
        print(precision)
    return precision >= seuil


def erreur_minimale_attendue(prediction, y_true, seuil, type_erreur='mse', affiche=0):
    """
    Calcule l'erreur minimale attendue selon le type d'erreur spécifié et vérifie si elle est en dessous du seuil.

    Arguments:
    - prediction (np.array): Les prédictions du modèle sous forme de probabilités.
    - y_true (np.array): Les étiquettes réelles.
    - seuil (float): Le seuil d'erreur à atteindre.
    - type_erreur (str): Le type d'erreur à calculer. Peut être 'mse' (Erreur Quadratique Moyenne),
                         'mae' (Erreur Absolue Moyenne), ou 'rae' (Erreur Relative Absolue).

    Retour:
    - bool: True si l'erreur calculée est en dessous du seuil, sinon False.
    """
    if type_erreur not in ['mse', 'mae', 'rae']:
        raise ValueError("type_erreur doit être 'mse', 'mae' ou 'rae'.")

    
    y_true_one_hot = np.eye(prediction.shape[1])[y_true]

    if type_erreur == 'mse':
        # Calculer l'erreur quadratique moyenne
        erreur = np.mean((prediction - y_true_one_hot) ** 2)
    elif type_erreur == 'mae':
        # Calculer l'erreur absolue moyenne
        erreur = np.mean(np.abs(prediction - y_true_one_hot))
    elif type_erreur == 'rae':
        # Transformer les probabilités en étiquettes de classe
        prediction_classes = np.argmax(prediction, axis=1)
        # Calculer l'erreur absolue moyenne
        abs_error = np.mean(np.abs(prediction_classes - y_true))
        # Calculer la baseline (erreur absolue moyenne par rapport à la moyenne des étiquettes réelles)
        baseline_error = np.mean(np.abs(np.mean(y_true) - y_true))
        # Calculer l'erreur relative absolue
        erreur = abs_error / baseline_error
    # Vérifier si cette erreur est en dessous du seuil
    if affiche:
        print(erreur)
    return erreur <= seuil


def stabilite_predictions(prediction_actuelle, prediction_precedente, seuil, affiche=0):
    """
    Vérifie la stabilité des prédictions entre deux itérations successives.

    Arguments:
    - prediction_actuelle (np.array): Les prédictions actuelles du modèle.
    - prediction_precedente (np.array): Les prédictions précédentes du modèle.
    - seuil (float): Le seuil de stabilité à atteindre (exprimé en pourcentage de prédictions stables).

    Retour:
    - bool: True si la stabilité est en dessous du seuil, sinon False.
    """
    # Calculer le pourcentage de prédictions stables
    prediction_actuelle = prediction_actuelle[:prediction_precedente.shape[0]]
    
    # Calculer le pourcentage de prédictions stables
    stabilite = np.mean(abs(prediction_actuelle - prediction_precedente))
    # Vérifier si cette stabilité est supérieure au seuil
    if affiche:
        print(stabilite)
    
    return stabilite <= seuil


def stabilite_predictions_2(prediction_actuelle, prediction_precedente, prediction_precedente_2, seuil, affiche=0):
    """
    Vérifie la stabilité des prédictions entre deux itérations successives.

    Arguments:
    - prediction_actuelle (np.array): Les prédictions actuelles du modèle.
    - prediction_precedente (np.array): Les prédictions précédentes du modèle.
    - seuil (float): Le seuil de stabilité à atteindre (exprimé en pourcentage de prédictions stables).

    Retour:
    - bool: True si la stabilité est en dessous du seuil, sinon False.
    """
    # Calculer le pourcentage de prédictions stables
    prediction_actuelle = prediction_actuelle[:prediction_precedente.shape[0]]
    
    
    # Calculer le pourcentage de prédictions stables
    stabilite = np.mean(abs(prediction_actuelle - prediction_precedente))
    
    prediction_precedente = prediction_precedente[:prediction_precedente_2.shape[0]]
    
    stabilite_2 = np.mean(abs(prediction_precedente - prediction_precedente_2))
    # Vérifier si cette stabilité est supérieure au seuil
    if affiche:
        print((stabilite + stabilite_2)/2)
    
    return (stabilite + stabilite_2) <= (2 * seuil)
