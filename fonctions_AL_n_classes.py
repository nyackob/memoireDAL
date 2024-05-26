"""
Script contenant les differentes fonctions principales d'encapsulation du code
""" 



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
import random
import math as m
import time
from informativite import *
from sklearn.svm import SVC
from representativite import *

def afficher_frontiere_decision(xx, yy, Z, X, X_labeled, y_labeled, title, n_classes):
    plt.figure(1)
    plt.clf()
    
    couleurs = plt.cm.tab10.colors

    pale_colors = [(r, g, b, 0.2) for (r, g, b) in couleurs[:n_classes]]
    cmap = ListedColormap(pale_colors)
    
    plt.contourf(xx, yy, Z, levels=np.arange(-0.5, n_classes, 1), cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], color='black', label="unlabeled")
    
    # Afficher les points labellisés sauf le dernier point ajouté
    handles = []
    labels = []
    for i in range(len(X_labeled) - 1):
        classe = y_labeled[i]
        scatter = plt.scatter(X_labeled[i][0], X_labeled[i][1], color=couleurs[classe % len(couleurs)])
        if f"Classe {classe}" not in labels:
            handles.append(scatter)
            labels.append(f"Classe {classe}")
    
    # Afficher le dernier point labellisé en jaune avec bordure colorée selon sa classe
    scatter = plt.scatter(X_labeled[-1][0], X_labeled[-1][1], color='yellow', edgecolor=couleurs[y_labeled[-1] % len(couleurs)])
    if f"Classe {y_labeled[-1]}" not in labels:
        handles.append(scatter)
        labels.append(f"Dernier point ajouté (Classe {y_labeled[-1]})")
    
    # Trier les handles et les labels par ordre croissant des labels
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: int(x[1].split()[-1]))
    handles, labels = zip(*sorted_handles_labels)
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(handles=handles, labels=labels)
    plt.show()
    
    
def Labelisation_de_depart(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_labelise_depart, nb_pts_par_classe, tol, pts_labelise_au_hasard, representation_mixte, part_de_label_par_densite, n_classes):
    X_labeled = X_labeled.tolist()
    y_labeled = y_labeled.tolist()
    counts = [0] * n_classes


    if pts_labelise_au_hasard == 0: 
        while any(count < nb_pts_par_classe for count in counts) or any(abs(counts[i] - counts[j]) > tol for i in range(n_classes) for j in range(i+1, n_classes)):

            distances = cdist(X_unlabeled, X_unlabeled, 'euclidean')

            if representation_mixte == 1:
                if sum(counts) <= (nb_pts_labelise_depart * part_de_label_par_densite) // 1:
                    u = selec_par_densite(X_unlabeled)
                else:
                    u = [ -x for x in selec_par_densite(X_unlabeled)]
            else: 
                u = selec_par_densite(X_unlabeled)

            query_idx = np.argmax(u)
            X_labeled.append(X_unlabeled[query_idx])
            y_labeled.append(y_unlabeled[query_idx])
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

            counts[y_labeled[-1]] += 1

    else:
        while any(count < nb_pts_par_classe for count in counts) or any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):

            query_idx = random.randint(0, len(X_unlabeled) - 1)
            counts[y_unlabeled[query_idx]] += 1

            if any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):
                counts[y_unlabeled[query_idx]] -= 1
            else:
                X_labeled.append(X_unlabeled[query_idx])
                y_labeled.append(y_unlabeled[query_idx])
                X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
                y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    return np.array(X_labeled), np.array(y_labeled), X_unlabeled, y_unlabeled, counts

def Labelisation_de_depart_diversite(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, tol, pts_labelise_au_hasard, n_classes):
    X_labeled = X_labeled.tolist()
    y_labeled = y_labeled.tolist()
    counts = [0] * n_classes
    if len(X_labeled)==0:
        query_idx = 0
    
        X_labeled.append(X_unlabeled[query_idx])
        y_labeled.append(y_unlabeled[query_idx])
        X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
        y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)
    if pts_labelise_au_hasard == 0: 
        while any(count < nb_pts_par_classe for count in counts) or any(abs(counts[i] - counts[j]) > tol for i in range(n_classes) for j in range(i+1, n_classes)):

            u = selec_par_diversite(X_unlabeled, X_labeled, method='distance')

            query_idx = np.argmax(u)
            X_labeled.append(X_unlabeled[query_idx])
            y_labeled.append(y_unlabeled[query_idx])
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

            counts[y_labeled[-1]] += 1

    else:
        while any(count < nb_pts_par_classe for count in counts) or any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):

            query_idx = random.randint(0, len(X_unlabeled) - 1)
            counts[y_unlabeled[query_idx]] += 1

            if any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):
                counts[y_unlabeled[query_idx]] -= 1
            else:
                X_labeled.append(X_unlabeled[query_idx])
                y_labeled.append(y_unlabeled[query_idx])
                X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
                y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    return np.array(X_labeled), np.array(y_labeled), X_unlabeled, y_unlabeled, counts



def Labelisation_de_depart_div(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_labelise_depart, pts_labelise_au_hasard):
    X_labeled = X_labeled.tolist()
    y_labeled = y_labeled.tolist()
    count = 1 
    
    query_idx = random.randint(0, len(X_unlabeled) - 1)

    X_labeled.append(X_unlabeled[query_idx])
    y_labeled.append(y_unlabeled[query_idx])
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    if pts_labelise_au_hasard == 0: 
        while (count < nb_pts_labelise_depart):
            
            u = selec_par_diversite(X_unlabeled, X_labeled, method='distance')

            query_idx = np.argmax(u)
            X_labeled.append(X_unlabeled[query_idx])
            y_labeled.append(y_unlabeled[query_idx])
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)
            print("here")
            count += 1

    else:
        while (count < nb_pts_labelise_depart):

            query_idx = random.randint(0, len(X_unlabeled) - 1)
            
            count += 1

            X_labeled.append(X_unlabeled[query_idx])
            y_labeled.append(y_unlabeled[query_idx])
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    counts = [count]
    return np.array(X_labeled), np.array(y_labeled), X_unlabeled, y_unlabeled, counts



def Labelisation_de_depart_clustering(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, pts_labelise_au_hasard, n_classes, method = 'kmeans'):
    X_labeled = X_labeled.tolist()
    y_labeled = y_labeled.tolist()
    n_clusters = nb_pts_par_classe * n_classes
    counts = [0] * n_classes

    if pts_labelise_au_hasard == 0: 
        
        U = selec_par_cluster_2(X_unlabeled, method, n_clusters, eps=0.5, min_samples = 10)
        
        for i in range(n_clusters):

            # Trouver l'indice de la valeur maximale
            query_idx = np.argmax(U)
            X_labeled.append(X_unlabeled[query_idx])
            y_labeled.append(y_unlabeled[query_idx])
            X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)
            del U[query_idx]

            counts[y_labeled[-1]] += 1

    else:
        while any(count < nb_pts_par_classe for count in counts) or any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):

            query_idx = random.randint(0, len(X_unlabeled) - 1)
            counts[y_unlabeled[query_idx]] += 1

            if any(abs(counts[i] - counts[j]) > 1 for i in range(n_classes) for j in range(i+1, n_classes)):
                counts[y_unlabeled[query_idx]] -= 1
            else:
                X_labeled.append(X_unlabeled[query_idx])
                y_labeled.append(y_unlabeled[query_idx])
                X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
                y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    return np.array(X_labeled), np.array(y_labeled), X_unlabeled, y_unlabeled, counts


def gerer_affichage_et_mise_a_jour(k, retard_frontiere, prev_xx, prev_yy, prev_Z, xx, yy, Z, X, X_labeled, y_labeled, avectemps, temps, N, n_classes):
    if k > 1 and retard_frontiere:
        if prev_xx is not None and prev_yy is not None and prev_Z is not None:
            afficher_frontiere_decision(prev_xx, prev_yy, prev_Z, X, X_labeled, y_labeled, f'Nouvelle frontière de décision N={k-1}', n_classes)
    
    prev_xx, prev_yy, prev_Z = xx, yy, Z

    if not retard_frontiere:
        afficher_frontiere_decision(xx, yy, Z, X, X_labeled, y_labeled, f'Nouvelle frontière de décision N={k}', n_classes)

    if avectemps == 1:
        plt.pause(temps)
    
    if k == N:
        return None, None, None

    return prev_xx, prev_yy, prev_Z

def mettre_a_jour_etiquettes(X_unlabeled, y_unlabeled, X_labeled, y_labeled, indice_max):
    y_labeled = np.append(y_labeled, y_unlabeled[indice_max])
    nouvel_echantillon = X_unlabeled[indice_max].reshape(1, -1)
    X_labeled = np.concatenate((X_labeled, nouvel_echantillon), axis=0)
    X_unlabeled = np.delete(X_unlabeled, indice_max, axis=0)
    y_unlabeled = np.delete(y_unlabeled, indice_max, axis=0)

    return X_unlabeled, y_unlabeled, X_labeled, y_labeled

def création_et_affichage_modele_theorique(modele_de_controle, Nbpts, num_random_state, n_classes, ann=0):
    plt.figure(2)
    plt.clf()

    X, y = make_blobs(n_samples=Nbpts, centers=n_classes, random_state=num_random_state)
    reel_X, reel_y = X, y
    
    if ann:
        # Entraîner le modèle ANN
        model2 = modele_de_controle
        model2.fit(reel_X, reel_y, epochs=50, batch_size=32, verbose=0)
        
        x_min, x_max = reel_X[:, 0].min() - 1, reel_X[:, 0].max() + 1
        y_min, y_max = reel_X[:, 1].min() - 1, reel_X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        # Prédire les étiquettes pour chaque point de la grille
        Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
    else: 
        model2 = modele_de_controle
        model2.fit(reel_X, reel_y)
    
        x_min, x_max = reel_X[:, 0].min() - 1, reel_X[:, 0].max() + 1
        y_min, y_max = reel_X[:, 1].min() - 1, reel_X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    
        Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
    couleurs = plt.cm.tab10.colors
    

    pale_colors = [(r, g, b, 0.2) for (r, g, b) in couleurs[:n_classes]]  
    cmap = ListedColormap(pale_colors)
    

    plt.contourf(xx, yy, Z, levels=np.arange(-0.5, n_classes, 1), cmap=cmap)
        

    for i in range(n_classes):  # Boucle sur n classes
        mask = y == i
        plt.scatter(reel_X[mask, 0], reel_X[mask, 1], color=couleurs[i], label=f'Classe {i}')

    plt.legend()
    plt.title('Données réelles : frontière de décision')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    return model2

def afficher_parametres_et_resultats(Nbpts, pts_labelise_au_hasard, representation_mixte, part_de_label_par_densite, nb_pts_par_classe, N, prediction_model, prediction_model_labeled, prediction_reel, y, y_labeled, n_classes):
    print("Nombre de points du pool : ", Nbpts)

    if pts_labelise_au_hasard != 0:
        print("Type de sélection des points de départ : Par Hasard")
    else:
        print("Type de sélection des points de départ : Par Représentation")

    print("Nombre de points labélisés au départ : ", n_classes * nb_pts_par_classe)
    print("Nombre de points étiquetés par information : ", N)
    print("Nombre de points étiquetés au total : ", n_classes * nb_pts_par_classe + N)

    concordance = np.mean(prediction_model == prediction_reel)
    print("Concordance entre les deux modèles :", concordance)
    
    pref_sur_donnée_labeled = np.mean(prediction_model_labeled == y_labeled)
    print("performance sur les données étiquetées :", pref_sur_donnée_labeled)
    
    pref_sur_donnée_reelle = np.mean(prediction_model == y)
    print("performance sur les données réelles :", pref_sur_donnée_reelle)
    
    pref_sur_modele_reelle = np.mean(prediction_reel == y)
    print("performance du modèle complet sur les données réelles :", pref_sur_modele_reelle)

