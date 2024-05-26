"""
Script contenant les fonctions d'utilitées des differentes méthodes de stratégies
de requêtes pour les stratégies basées sur la représentation
""" 




import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.stats import entropy
from sklearn.cluster import KMeans, DBSCAN
from joblib import parallel_backend


def selec_par_densite(X_unlabeled, method='distance', gamma1=1, gamma2=0.5, alpha=1):
    """
    Calcule le score d'utilité pour chaque point non étiqueté basé sur des méthodes de densité.
    
    Arguments:
        X_unlabeled (np.array): Données non étiquetées.
        method (str): Méthode de calcul de la densité ('distance', 'cos', 'KL', 'gauss').
        gamma1 (float): Paramètre de lissage pour la divergence KL.
        gamma2 (float): Paramètre de lissage pour la divergence KL.
        alpha (float): Variance pour la similarité gaussienne.
    
    Retour:
        np.array: Scores d'utilité pour chaque point non étiqueté.
    """
    if method not in ['distance', 'cos', 'KL', 'gauss']:
        raise ValueError("Method must be 'distance', 'cos', 'KL', or 'gauss'")
    
    U = []
    U_size = len(X_unlabeled)
    
    if method == 'distance':
        for i, x_u in enumerate(X_unlabeled):
            distances = np.linalg.norm(X_unlabeled[np.arange(U_size) != i] - x_u, axis=1)
            min_distance = np.min(distances)
            U.append(-(min_distance + 1e-9))
    
    elif method == 'cos':
        for i, x_u in enumerate(X_unlabeled):
            similarities = cosine_similarity([x_u], X_unlabeled[np.arange(U_size) != i])[0]
            mean_similarity = np.mean(similarities)
            U.append(mean_similarity)
    
    elif method == 'KL':
        for i, x_u in enumerate(X_unlabeled):
            # Estimer la distribution locale autour de chaque point non étiqueté
            p_x_u = np.mean(X_unlabeled[np.arange(U_size) != i], axis=0)
            # Estimer la distribution globale des points non étiquetés
            p_global = np.mean(X_unlabeled, axis=0)
            # Calculer la divergence KL entre les distributions
            kl_div = np.sum(p_x_u * np.log(p_x_u / (gamma2 * p_global + (1 - gamma2) * p_x_u + 1e-9)))
            kl_sim = np.exp(-gamma1 * kl_div)
            U.append(kl_sim)
    
    elif method == 'gauss':
        for i, x_u in enumerate(X_unlabeled):
            gauss_sim = np.exp(-np.sum((X_unlabeled[np.arange(U_size) != i] - x_u) ** 2, axis=1) / (2 * alpha ** 2))
            mean_gauss_sim = np.mean(gauss_sim)
            U.append(mean_gauss_sim)
            
    return U



def selec_par_diversite(X_unlabeled, X_labeled, method='distance'):
    """
    Calcule les scores d'utilité pour chaque point non étiqueté basé sur des méthodes de diversité.
    
    Arguments:
        X_unlabeled (np.array): Données non étiquetées.
        X_labeled (np.array): Données étiquetées.
        method (str): Méthode de diversité ('distance' pour la distance euclidienne, 'cos' pour la similarité cosinus).
    
    Retour:
        np.array: Scores d'utilité pour chaque point non étiqueté.
    """
    U_size = len(X_unlabeled)
    L_size = len(X_labeled)
    
    U=[]
    
    if method == 'distance':
        for i, x_u in enumerate(X_unlabeled):
            distances = np.linalg.norm(X_labeled - x_u, axis=1)
            min_distance = np.min(distances)
            U.append(min_distance)
    
    elif method == 'cos':
        for i, x_u in enumerate(X_unlabeled):
            similarities = cosine_similarity([x_u], X_labeled)[0]
            mean_similarity = np.mean(similarities)
            U.append(-mean_similarity)
    return U


def selec_par_cluster(X_unlabeled, method='kmeans', n_clusters=10, eps=0.5, min_samples=5):
    """
    Calcule les scores d'utilité pour chaque point non étiqueté basé sur des méthodes de clustering.
    
    Arguments:
        X_unlabeled (np.array): Données non étiquetées.
        method (str): Méthode de clustering ('kmeans' ou 'dbscan').
        n_clusters (int): Nombre de clusters pour K-means.
        eps (float): Paramètre epsilon pour DBSCAN.
        min_samples (int): Nombre minimum d'échantillons dans un voisinage pour DBSCAN.
    
    Retour:
        np.array: Scores d'utilité pour chaque point non étiqueté.
    """
    U = []
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_unlabeled)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        for i, x_u in enumerate(X_unlabeled):
            cluster_center = cluster_centers[labels[i]]
            distance_to_center = np.linalg.norm(x_u - cluster_center)
            U.append(-distance_to_center) # Utilité négative de la distance (plus proche du centre = plus utile)
    
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X_unlabeled)
        labels = dbscan.labels_
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Ignorer les points bruyants 
            cluster_points = X_unlabeled[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            
            for i, x_u in enumerate(X_unlabeled):
                if labels[i] == label:
                    distance_to_center = np.linalg.norm(x_u - cluster_center)
                    U.append(-distance_to_center)  # Utilité négative de la distance (plus proche du centre = plus utile)
   
    return U

