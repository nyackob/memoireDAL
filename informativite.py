"""
Script contenant les fonctions d'utilitées des differentes méthodes de stratégies
de requêtes pour les stratégies basées sur l'information
""" 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import entropy
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
# Exemple de modèle de prédiction fictif

def incertitude_moindre_confiance(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection, ann=0):
    """
    Calcule l'incertitude de la prédiction pour X_unlabeled en utilisant la mesure de moindre confiance.

    La fonction `u(x, h)` est définie comme étant 1 moins la probabilité prédite pour la classe la plus probable
    pour l'instance `x` par le modèle `h`. En pratique, nous passons en paramètre les prédictions plutôt que le 
    modèle directement.

    Arguments:
        X_unlabeled (np.array): Les caractéristiques des exemples pour lesquels on veut calculer l'incertitude.
        prediction (np.array): Les probabilités prédites pour chaque classe par le modèle pour chaque exemple 
                               de X_unlabeled. Chaque élément de prediction est un tableau de probabilités.

    Retour:
        list: Une liste contenant l'incertitude de la prédiction pour chaque exemple de X_unlabeled.

    Exemple:
        >>> import numpy as np
        >>> X_unlabeled = np.array([[1, 2, 3], [4, 5, 6]])  # Exemples non étiquetés
        >>> prediction = np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])  # Probabilités prédites pour chaque exemple
        >>> incertitudes = incertitude_moindre_confiance(X_unlabeled, prediction)
        >>> print(incertitudes)
        [0.3, 0.6]
    """
    if ann:
        # Entraîner le modèle ANN
        model_selec = modele_selection
        model_selec.fit(X_labeled, y_labeled, epochs=50, batch_size=32, verbose=0)
    
        # Prédire les étiquettes pour chaque point de la grille
        Z = model_selec.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
        prediction_selc = model_selec.predict(X_unlabeled)
    else:
        model_selec = modele_selection
        model_selec.fit(X_labeled, y_labeled)
        
        # Prédire les étiquettes pour chaque point de la grille
        Z = model_selec.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # On demande les probas de chaque point non etiqueté
        prediction_selc = model_selec.predict_proba(X_unlabeled)

    
    U = []
    for i in range(len(X_unlabeled)):
        # Trouver la probabilité de la classe la plus probable
        probabilite_maximale = max(prediction_selc[i])
        
        # Calculer l'incertitude
        incertitude = 1 - probabilite_maximale
        
        # Ajouter l'incertitude calculée à la liste U
        U.append(incertitude)
    
    return U

def incertitude_moindre_confiance_ann(X_unlabeled, prediction_selc):
    """
    Calcule l'incertitude de la prédiction pour X_unlabeled en utilisant la mesure de moindre confiance.

    La fonction `u(x, h)` est définie comme étant 1 moins la probabilité prédite pour la classe la plus probable
    pour l'instance `x` par le modèle `h`. En pratique, nous passons en paramètre les prédictions plutôt que le 
    modèle directement.

    Arguments:
        X_unlabeled (np.array): Les caractéristiques des exemples pour lesquels on veut calculer l'incertitude.
        prediction (np.array): Les probabilités prédites pour chaque classe par le modèle pour chaque exemple 
                               de X_unlabeled. Chaque élément de prediction est un tableau de probabilités.

    Retour:
        list: Une liste contenant l'incertitude de la prédiction pour chaque exemple de X_unlabeled.

    Exemple:
        >>> import numpy as np
        >>> X_unlabeled = np.array([[1, 2, 3], [4, 5, 6]])  # Exemples non étiquetés
        >>> prediction = np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])  # Probabilités prédites pour chaque exemple
        >>> incertitudes = incertitude_moindre_confiance(X_unlabeled, prediction)
        >>> print(incertitudes)
        [0.3, 0.6]
    """
    
    U = []
    for i in range(len(X_unlabeled)):
        # Trouver la probabilité de la classe la plus probable
        probabilite_maximale = max(prediction_selc[i])
        
        # Calculer l'incertitude 
        incertitude = 1 - probabilite_maximale
        
        # Ajouter l'incertitude calculée à la liste U
        U.append(incertitude)
    
    return U




def incertitude_entropie(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection):
    """
    Calcule l'incertitude de la prédiction pour X_unlabeled en utilisant la mesure par l'entropie.

    La fonction `u(x, h)` est définie comme étant l'entropie des probabilités prédites pour les classes 
    pour l'instance `x` par le modèle `h`. En pratique, nous passons en paramètre les prédictions plutôt que le 
    modèle directement.

    Arguments:
        X_unlabeled (np.array): Les caractéristiques des exemples pour lesquels on veut calculer l'incertitude.
        prediction (np.array): Les probabilités prédites pour chaque classe par le modèle pour chaque exemple 
                               de X_unlabeled. Chaque élément de prediction est un tableau de probabilités.

    Retour:
        list: Une liste contenant l'incertitude de la prédiction pour chaque exemple de X_unlabeled.

    Exemple:
        >>> import numpy as np
        >>> X_unlabeled = np.array([[1, 2, 3], [4, 5, 6]])  # Exemples non étiquetés
        >>> prediction = np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])  # Probabilités prédites pour chaque exemple
        >>> incertitudes = incertitude_entropie(X_unlabeled, prediction)
        >>> print(incertitudes)
        [0.8018185525433374, 1.0888999753452238]
    """
    model_selec = modele_selection
    model_selec.fit(X_labeled, y_labeled)
   
    # Prédire les étiquettes pour chaque point de la grille
    Z = model_selec.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # On demande les probas de chaque point non etiqueté
    prediction_selc = model_selec.predict_proba(X_unlabeled)
    
    U = []
    for i in range(len(X_unlabeled)):
        
        # Calculer l'entropie
        entropie = -np.sum(prediction_selc[i] * np.log(prediction_selc[i] + 1e-9))  # On ajoute un petit epsilon pour éviter log(0)
        
        # Ajouter l'entropie calculée à la liste U
        U.append(entropie)
    
    return U


def incertitude_marges(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection):
    """
    Calcule l'incertitude de la prédiction pour X_unlabeled en utilisant la mesure par échantillonnage avec marges.

    La fonction `u(x, h)` est définie comme étant l'inverse de la différence entre les probabilités prédictes pour
    la classe la plus probable et la deuxième classe la plus probable pour l'instance `x` par le modèle `h`.
    En pratique, nous passons en paramètre les prédictions plutôt que le modèle directement.

    Arguments:
        X_unlabeled (np.array): Les caractéristiques des exemples pour lesquels on veut calculer l'incertitude.
        prediction (np.array): Les probabilités prédites pour chaque classe par le modèle pour chaque exemple
                               de X_unlabeled. Chaque élément de prediction est un tableau de probabilités.

    Retour:
        list: Une liste contenant l'incertitude de la prédiction pour chaque exemple de X_unlabeled.

    Exemple:
        >>> import numpy as np
        >>> X_unlabeled = np.array([[1, 2, 3], [4, 5, 6]])  # Exemples non étiquetés
        >>> prediction = np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])  # Probabilités prédites pour chaque exemple
        >>> incertitudes = incertitude_marges(X_unlabeled, prediction)
        >>> print(incertitudes)
        [1.4285714285714286, 2.5]
    """
    model_selec = modele_selection
    model_selec.fit(X_labeled, y_labeled)
   
    # Prédire les étiquettes pour chaque point de la grille
    Z = model_selec.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # On demande les probas de chaque point non etiqueté
    prediction_selc = model_selec.predict_proba(X_unlabeled)
    U = []
    for i in range(len(X_unlabeled)):
        # Obtenir les probabilités prédites pour l'exemple i
        probabilites = prediction_selc[i]
        
        # Trouver les deux plus grandes probabilités
        probabilites_tries = np.sort(probabilites)[::-1]  # Tri décroissant
        P1 = probabilites_tries[0]  # Probabilité de la classe la plus probable
        P2 = probabilites_tries[1]  # Probabilité de la deuxième classe la plus probable
        
        # Calculer l'incertitude
        incertitude = -(P1 - P2)
        
        # Ajouter l'incertitude calculée à la liste U
        U.append(incertitude)
    
    return U


def requete_par_comite(X_unlabeled, X_labeled, y_labeled, modele, mesure='vote_majoritaire', n_comite=5):
    """
    Calcule l'incertitude des prédictions pour X_unlabeled en utilisant la méthode de requête par comité
    avec différentes mesures de désaccord.

    La méthode de requête par comité consiste à utiliser plusieurs modèles formés sur des sous-ensembles 
    aléatoires des données étiquetées et à évaluer chaque instance non étiquetée. Les instances générant le plus 
    de désaccord entre les modèles sont considérées comme les plus utiles.

    Arguments:
        X_unlabeled (np.array): Les caractéristiques des exemples non étiquetés pour lesquels on veut calculer l'incertitude.
        X_labeled (np.array): Les caractéristiques des exemples étiquetés.
        y_labeled (np.array): Les étiquettes des exemples étiquetés.
        modele (class): La classe du modèle à utiliser pour l'entraînement (par exemple, LogisticRegression).
        mesure (str): La méthode de mesure de désaccord ('vote_majoritaire', 'entropie', 'kullback_leibler').
        n_comite (int): Le nombre de modèles dans le comité (par défaut 5).

    Retour:
        list: Une liste contenant l'incertitude de la prédiction pour chaque exemple de X_unlabeled.

    Exemple:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X_labeled = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        >>> y_labeled = np.array([0, 1, 0, 1, 0])
        >>> X_unlabeled = np.array([[6, 7], [7, 8], [8, 9]])
        >>> incertitudes = requete_par_comite(X_unlabeled, X_labeled, y_labeled, LogisticRegression, mesure='entropie')
        >>> print(incertitudes)
    """
    # Liste pour stocker les incertitudes
    U = []
    
    if str(modele) == "LogisticRegression()":
        modele = LogisticRegression

    # Créer un comité de modèles
    comite = []
    for _ in range(n_comite):
        # Créer un échantillon bootstrap des données étiquetées
        X_resample, y_resample = resample(X_labeled, y_labeled)
        # print(X_resample)
        # Entraîner un modèle sur l'échantillon bootstrap
        model = modele()
        model.fit(X_resample, y_resample)
        comite.append(model)
    # Prédire les probabilités pour chaque exemple non étiqueté avec chaque modèle du comité
    for x in X_unlabeled:
        predictions = []
        for model in comite:
            # x = np.reshape(x, (1, -1))
            # predictions.append(model.predict(x)[0]) # Probabilités pour l'exemple x
            predictions.append(model.predict_proba([x])[0]) 
        # Convertir les prédictions en un tableau numpy pour un traitement plus facile
        predictions = np.array(predictions)

        # Calculer l'incertitude en fonction de la méthode de mesure de désaccord
        if mesure == 'vote_majoritaire':
            # Vote majoritaire
            predictions_labels = np.argmax(predictions, axis=1)
            counts = np.bincount(predictions_labels)
            majoritaire = np.argmax(counts)
            incertitude = 1 - (counts[majoritaire] / n_comite)

        elif mesure == 'entropie':
            # Entropie de Shannon
            moyenne_predictions = np.mean(predictions, axis=0)
            incertitude = entropy(moyenne_predictions)

        elif mesure == 'kullback_leibler':
            # Divergence de Kullback-Leibler
            moyenne_predictions = np.mean(predictions, axis=0)
            incertitude = np.mean([entropy(p, moyenne_predictions) for p in predictions])
            
        elif mesure == 'variance':
            # Variance des prédictions
            variance_predictions = np.var(predictions, axis=0)
            incertitude = np.mean(variance_predictions)

        elif mesure == 'distance_euclidienne':
            # Distance euclidienne entre les prédictions
            moyenne_predictions = np.mean(predictions, axis=0)
            incertitude = np.mean([np.linalg.norm(p - moyenne_predictions) for p in predictions])
            
        else:
            raise ValueError("Mesure de désaccord non reconnue. Utilisez 'vote_majoritaire', 'entropie' ou 'kullback_leibler'.")

        # Ajouter l'incertitude calculée à la liste U
        U.append(incertitude)
    
    return U


import numpy as np
from sklearn.linear_model import LogisticRegression

def changement_modele_attendu_pour_RL(X_unlabeled, X_labeled, y_labeled, xx, yy, methode='EGL'):
    """
    Calcule le score d'utilité pour le Changement de Modèle Attendu selon la méthode spécifiée.

    Arguments:
        model (LogisticRegression): Le modèle utilisé pour calculer le gradient ou le changement de poids.
        X_unlabeled (np.array): Les caractéristiques des exemples non étiquetés.
        y_prob (np.array): Les probabilités prédites pour chaque classe par le modèle pour chaque exemple.
        methode (str): La méthode à utiliser pour le calcul ('EGL' ou 'EWC').

    Retour:
        list: Une liste contenant le score d'utilité pour chaque exemple non étiqueté.
    """
    model_selec = LogisticRegression()
    model_selec.fit(X_labeled, y_labeled)
   
    # Prédire les étiquettes pour chaque point de la grille
    Z = model_selec.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # On demande les probas de chaque point non etiqueté
    prediction_selc = model_selec.predict_proba(X_unlabeled)
    
    if methode not in ['EGL', 'EWC']:
        raise ValueError("La méthode doit être 'EGL' ou 'EWC'.")

    U = []
    for i in range(len(X_unlabeled)):
        score_sum = 0
        for c in range(prediction_selc.shape[1]):  # Pour chaque classe
            y_c = np.zeros(prediction_selc.shape[1])
            y_c[c] = 1
            # Utiliser le produit externe pour obtenir les dimensions correctes
            loss_gradient = np.outer(prediction_selc[i, c] - y_c, X_unlabeled[i])
            
            if methode == 'EGL':
                score = np.linalg.norm(loss_gradient)
            elif methode == 'EWC':
                score = np.sum(np.abs(loss_gradient))
            
            score_sum += score
        U.append(score_sum)
    return U



def changement_erreur_attendu(modele, X_labeled, y_labeled, X_unlabeled, y_unlabeled):
    """
    Calcule le score d'utilité pour chaque point non étiqueté basé sur la méthode de Changement d'Erreur/Prédiction Attendu.
    
    Arguments:
        model (LogisticRegression): Le modèle initial.
        X_labeled (np.array): Données étiquetées.
        y_labeled (np.array): Étiquettes des données étiquetées.
        X_unlabeled (np.array): Données non étiquetées.
        y_unlabeled (np.array): Étiquettes des données non étiquetées (seulement pour évaluation).
    
    Retour:
        U: Scores d'utilité pour chaque point non étiqueté.
    """
    U = []

    for i in range(len(X_unlabeled)):
        X_temp = np.vstack((X_labeled, X_unlabeled[i].reshape(1, -1)))
        y_temp = np.append(y_labeled, y_unlabeled[i])
        
        model_temp = modele
        model_temp.fit(X_temp, y_temp)
        
        # Calculer l'erreur future attendue sur les données non étiquetées restantes
        y_prob = model_temp.predict_proba(X_unlabeled)
        expected_error = np.mean([log_loss([y], [prob], labels=model_temp.classes_) for y, prob in zip(y_unlabeled, y_prob)])
        
        U.append(expected_error)

    return U


def svm_marge_maxmin(X_unlabeled, X_train, y_train, modele_selection):
    """
    Calcule les scores d'utilité des points non étiquetés selon la méthode de marge MaxMin.
    
    Arguments:
    X_unlabeled (numpy array): Ensemble des points non étiquetés.
    X_train (numpy array): Ensemble des points étiquetés (features).
    y_train (numpy array): Ensemble des points étiquetés (labels).
    modele_selection (SVC): Modèle SVM pour la sélection des points.

    Retour:
    numpy array: Tableau des scores d'utilité pour chaque point non étiqueté.
    """
    U = []
    
    for i in range(X_unlabeled.shape[0]):
        # Créer les ensembles de données temporaires
        X_temp_pos = np.vstack([X_train, X_unlabeled[i]])
        y_temp_pos = np.append(y_train, 1)
        
        X_temp_neg = np.vstack([X_train, X_unlabeled[i]])
        y_temp_neg = np.append(y_train, 0)
        
        # Entraîner le SVM en supposant le point appartenant à la classe positive
        svm_model_pos = SVC(kernel='linear', C=1.0)
        svm_model_pos.fit(X_temp_pos, y_temp_pos)
        support_vectors_pos = svm_model_pos.support_vectors_
        margin_pos = np.abs(svm_model_pos.decision_function(support_vectors_pos)).sum()
        
        # Entraîner le SVM en supposant le point appartenant à la classe négative
        svm_model_neg = SVC(kernel='linear', C=1.0)
        svm_model_neg.fit(X_temp_neg, y_temp_neg)
        support_vectors_neg = svm_model_neg.support_vectors_
        margin_neg = np.abs(svm_model_neg.decision_function(support_vectors_neg)).sum()
        
        # Prendre la somme des marges pour chaque classe
        min_margin = min(margin_pos, margin_neg)
        U.append(min_margin)
    
    return U

def svm_marge_ratio(X_unlabeled, X_train, y_train, modele_selection):
    """
    Calcule les scores d'utilité des points non étiquetés selon la méthode de marge MaxMin.
    
    Arguments:
    X_unlabeled (numpy array): Ensemble des points non étiquetés.
    X_train (numpy array): Ensemble des points étiquetés (features).
    y_train (numpy array): Ensemble des points étiquetés (labels).
    modele_selection (SVC): Modèle SVM pour la sélection des points.

    Retour:
    numpy array: Tableau des scores d'utilité pour chaque point non étiqueté.
    """
    U = []
    
    for i in range(X_unlabeled.shape[0]):
        # Créer les ensembles de données temporaires
        X_temp_pos = np.vstack([X_train, X_unlabeled[i]])
        y_temp_pos = np.append(y_train, 1)
        
        X_temp_neg = np.vstack([X_train, X_unlabeled[i]])
        y_temp_neg = np.append(y_train, 0)
        
        # Entraîner le SVM en supposant le point appartenant à la classe positive
        svm_model_pos = SVC(kernel='linear', C=1.0)
        svm_model_pos.fit(X_temp_pos, y_temp_pos)
        support_vectors_pos = svm_model_pos.support_vectors_
        margin_pos = np.abs(svm_model_pos.decision_function(support_vectors_pos)).sum()
        
        # Entraîner le SVM en supposant le point appartenant à la classe négative
        svm_model_neg = SVC(kernel='linear', C=1.0)
        svm_model_neg.fit(X_temp_neg, y_temp_neg)
        support_vectors_neg = svm_model_neg.support_vectors_
        margin_neg = np.abs(svm_model_neg.decision_function(support_vectors_neg)).sum()
        
        # Prendre la somme des marges pour chaque classe
        min_margin = min(margin_pos,margin_neg)/max(margin_pos, margin_neg)
        U.append(min_margin)
    
    return U
