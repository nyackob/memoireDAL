from informativite import *
from representativite import *
from fonctions_AL_n_classes import *
from sklearn.svm import SVC
from critere_arret import *
from joblib import parallel_backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# modèle ANN
def ann_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

ann=0

#------------- Parametrage -----------////////////////////////////////////////


# >> Données et paramètres principaux ---------------------------------------------------------------

num_random_state = 0 # numero de pool (interessant : 0, 4, 5, 6)

N = 100# Nombre d'iteration (nouvel etiquetage)

N_min = 2

Nbpts = 300

n_classes = 2 # Nombre de classes

taille_lot = 1

cluster = 0 # utilisation d'un modèle de clustering ?

nb_cluster = 10 # si oui Nb de cluster pour kmeans

rayon = 0.5 # si DBSCAN rayon ?

nb_voisins = 10 # si DBSCAN nb de voisins ?


# >> Affichage ------------------------------------------

# temps de pause entre chaque affichage d'itération  

retard_frontiere = 1 # i.e. afficher la frontiere de decision à partir de laquel le point en
                         # surbrilance à été séléctionné
avectemps=1 # oui=1 / non=0

temps= 1 # temps en seconde


#--------------------------------------------------------

# >> initialisation ------------------------------------

X_labeled = []
y_labeled = []

X_labeled = np.array(X_labeled)
y_labeled = np.array(y_labeled)

X, y = make_blobs(n_samples=Nbpts, n_features=2, centers=n_classes, random_state=num_random_state)
X_unlabeled, y_unlabeled = X, y



# >> Labelisation de départ----------------------------------------

nb_pts_labelise_depart = n_classes * 6 # Par exemple, 5 points par classe au départ

nb_pts_par_classe = nb_pts_labelise_depart // n_classes

pts_labelise_au_hasard=0 # 1 pour oui / 0 pour non

tol=3 #tolerance d'équart de point de chaque classe

representation_mixte=0 # mixe de stratégie par densité et par diversité (si == 0 par densité uniquement)

part_de_label_par_densite=0.5  # Valable ssi representation_mixte == 1


# -------   classique : densité / densité inversé  ----------------

# X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_labelise_depart, nb_pts_par_classe, tol, pts_labelise_au_hasard, representation_mixte, part_de_label_par_densite, n_classes)

# ------- diveristé (dist. euclidienne) --------------------------

# equilibré
# X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart_diversite(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, tol, pts_labelise_au_hasard, n_classes)

# Non equilibré
# X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart_div(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_labelise_depart, pts_labelise_au_hasard)

# ------------- Clustering --------------------------------------

# kmeans
# X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart_clustering(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, pts_labelise_au_hasard, n_classes)


# DBSCAN
# X_labeled, y_labeled, X_unlabeled, y_unlabeled, nb_pts_labelises = Labelisation_de_depart_clustering(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, pts_labelise_au_hasard, n_classes, 'dbscan')

#-------------- COMBINAISON -------------------------

# nb_pts_labelises = np.sum(tab_nb_pts_labelises)

nb_pts_par_classe = nb_pts_par_classe//2

# kmeans
X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart_clustering(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, pts_labelise_au_hasard, n_classes)
nb_pts_labelises = np.sum(tab_nb_pts_labelises)

# Diversité
X_labeled, y_labeled, X_unlabeled, y_unlabeled, tab_nb_pts_labelises = Labelisation_de_depart_diversite(X_unlabeled, y_unlabeled, X_labeled, y_labeled, nb_pts_par_classe, tol, pts_labelise_au_hasard, n_classes)
nb_pts_labelises += np.sum(tab_nb_pts_labelises)

# >> Modèles ----------------------------------------

# SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
# LogisticRegression()

#------ Regression Logistique -------------

modele_classifieur = LogisticRegression()

modele_de_controle = LogisticRegression()

modele_selection = modele_classifieur

#--------------- SVM --------------------

# modele_classifieur = SVC(kernel='linear', C=1, gamma='scale', probability=True)

# modele_de_controle = SVC(kernel='linear', C=1, gamma='scale', probability=True)

# modele_selection = SVC(kernel='linear', C=1, gamma='scale', probability=True)

#------ ANN -------------

# ann = 1

# taille_lot = 10

# modele_classifieur = ann_model(input_dim=X_unlabeled.shape[1], output_dim=n_classes)

# modele_de_controle = ann_model(input_dim=X_unlabeled.shape[1], output_dim=n_classes)

# modele_selection = modele_classifieur

# >> Premier entraîner du modèle du classifieur sur les points labellisés ----------

    # Créer une grille pour afficher la frontière de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

if ann:
    # Entraîner le modèle ANN
    model = modele_classifieur
    model.fit(X_labeled, y_labeled, epochs=50, batch_size=32, verbose=0)

    # Prédire les étiquettes pour chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    prediction_clas_lab = model.predict(X_labeled)
    prediction_clas_unlab = model.predict(X_unlabeled)
    
    prediction_clas_lab_prec = prediction_clas_lab 
    
else:
    
    model = modele_classifieur
    model.fit(X_labeled, y_labeled)
    
    # Prédire les étiquettes pour chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    prediction_clas_unlab = model.predict_proba(X_unlabeled)
    prediction_clas_lab = model.predict_proba(X_labeled)
    
    prediction_clas_lab_prec = prediction_clas_lab 
    

# >> Début de boucle principale avec pour condition d'arrêt un budget prédéfini --------

# Variables pour stocker l'état précédent de la grille et des prédictions
prev_xx, prev_yy, prev_Z = None, None, None

k=0

seuil = 0.015 #seuil d'arret 

while budget_predefini(k,N):
    

    # # >> critère s'arrêt -------------------------------------
    # if k >= N_min and incertitude_maximale(prediction_clas_lab, seuil):
    # if k >= N_min and incertitude_globale(prediction_clas_lab, seuil):
    # if k >= N_min and precision_selectionnee(prediction_clas_lab, y_labeled, seuil):
    # if k >= N_min and erreur_minimale_attendue(prediction_clas_lab, y_labeled, seuil, type_erreur='mse'):
    # if k >= N_min and stabilite_predictions(prediction_clas_lab, prediction_clas_lab_prec, seuil,1):
    if k > N_min and stabilite_predictions_2(prediction_clas_lab, prediction_clas_lab_prec, prediction_clas_lab_prec_2, seuil,1):
        break
    
    prediction_clas_lab_prec_2 = prediction_clas_lab_prec 
    # prediction du modèle de sélection sur les points labellisés
    prediction_clas_lab_prec = prediction_clas_lab 

    for j in range(taille_lot):
        k += 1
        prev_xx, prev_yy, prev_Z = gerer_affichage_et_mise_a_jour(k, retard_frontiere, prev_xx, prev_yy, prev_Z, xx, yy, Z, X, X_labeled, y_labeled, avectemps, temps, N, n_classes)
        
 # >> choix de la stratégie de requête
       
        if cluster:
            U = selec_par_cluster_2(X_unlabeled, 'kmeans', nb_cluster, rayon, nb_voisins)

            for i in range(nb_cluster):
                utilite_max = max(U)
                # Trouver l'indice de la valeur maximale
                indice_max = U.index(utilite_max)
                X_unlabeled, y_unlabeled, X_labeled, y_labeled = mettre_a_jour_etiquettes(X_unlabeled, y_unlabeled, X_labeled, y_labeled, indice_max)
                prev_xx, prev_yy, prev_Z = gerer_affichage_et_mise_a_jour(k, retard_frontiere, prev_xx, prev_yy, prev_Z, xx, yy, Z, X, X_labeled, y_labeled, avectemps, temps, N, n_classes)
                
                if prev_xx is None and prev_yy is None and prev_Z is None:
                    break
                del U[indice_max]
            break
        else:
            
            #--- Stratégie sur l'information ---------------
            
            
            # U = incertitude_moindre_confiance_ann(X_unlabeled, prediction_clas_unlab)
            
            # U = incertitude_moindre_confiance(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection)
            U = incertitude_entropie(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection)
            # U = incertitude_marges(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection)
            # U = requete_par_comite(X_unlabeled, X_labeled, y_labeled, modele_selection, 'entropie', 2)
            # U = changement_modele_attendu_pour_RL(X_unlabeled, X_labeled,y_labeled, xx, yy, methode='EWC')

            # U = svm_marge_maxmin(X_unlabeled, X_labeled, y_labeled, modele_selection)
            # U = svm_marge_ratio(X_unlabeled, X_labeled, y_labeled, modele_selection)
            
            #--- Stratégie sur l'information ---------------
            
            # U = selec_par_densite(X_unlabeled, 'distance')
            # U = selec_par_diversite(X_unlabeled, X_labeled, method='distance')
            # U = selec_par_cluster(X_unlabeled, 'kmeans', 10, 1, 10)

            #---  combinaison ------------------------
            # U1 = incertitude_moindre_confiance_ann(X_unlabeled, prediction_clas_unlab)
            # # U1 = incertitude_moindre_confiance(X_unlabeled, X_labeled, y_labeled, xx, yy, modele_selection)
            # U2 = selec_par_diversite(X_unlabeled, X_labeled, method='distance')
            # alpha = 5
            # beta = 1
            # U1_max = max(U1)
            # U2_max = max(U2)
            # U = [((a*10)/U1_max)**(alpha) * ((b*10)/U2_max)**(beta) for a, b in zip(U1, U2)]
            #-------------------------------------
            utilite_max = max(U)
            print(utilite_max)
            indice_max = U.index(utilite_max)
            # print(indice_max)
            X_unlabeled, y_unlabeled, X_labeled, y_labeled = mettre_a_jour_etiquettes(X_unlabeled, y_unlabeled, X_labeled, y_labeled, indice_max)
    
    if prev_xx is None and prev_yy is None and prev_Z is None:
        break
    
    if cluster:
        k += nb_cluster
        break

    if ann:
        # Entraîner le modèle ANN
        model = modele_classifieur
        model.fit(X_labeled, y_labeled, epochs=50, batch_size=32, verbose=0)
    
        # Prédire les étiquettes pour chaque point de la grille
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
        prediction_clas_lab = model.predict(X_labeled)
        prediction_clas_unlab = model.predict(X_unlabeled)
        
    else:
        
        model = modele_classifieur
        model.fit(X_labeled, y_labeled)
         
    
        
        # Prédire les étiquettes pour chaque point de la grille
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        prediction_clas_unlab = model.predict_proba(X_unlabeled)
        prediction_clas_lab = model.predict_proba(X_labeled)

    
if retard_frontiere == 1:
    k = k
    prev_xx, prev_yy, prev_Z = gerer_affichage_et_mise_a_jour(k, 0, prev_xx, prev_yy, prev_Z, xx, yy, Z, X, X_labeled, y_labeled, avectemps, temps, N, n_classes)

# Appel de la fonction pour afficher la frontière de décision finale
if ann:
    model2 = création_et_affichage_modele_theorique(modele_de_controle, Nbpts, num_random_state, n_classes,1)
else:
    model2 = création_et_affichage_modele_theorique(modele_de_controle, Nbpts, num_random_state, n_classes)

prediction_model = model.predict(X)

prediction_model_labeled = model.predict(X_labeled)

prediction_reel = model2.predict(X)


if ann:
    prediction_model =  np.argmax(prediction_model, axis=1)
    
    prediction_model_labeled =  np.argmax(prediction_model_labeled, axis=1)
    
    prediction_reel =  np.argmax(prediction_reel, axis=1)


nb_pts_labelises_total = np.sum(nb_pts_labelises)//n_classes



# Appel de la fonction pour afficher les paramètres et les résultats
afficher_parametres_et_resultats(Nbpts, pts_labelise_au_hasard, representation_mixte, part_de_label_par_densite, nb_pts_labelises_total, k, prediction_model, prediction_model_labeled, prediction_reel, y, y_labeled, n_classes)
