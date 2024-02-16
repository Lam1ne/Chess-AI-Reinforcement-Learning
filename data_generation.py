import numpy as np
import chess
from utils.chess_utils import determiner_resultat, coup_to_onehot, transformer_etat

# Fonction pour choisir un coup basé sur la politique donnée
def choisir_coup(plateau, politique):
    coups_possibles = list(plateau.legal_moves)  # Liste de tous les coups possibles
    index_coup_choisi = np.argmax(politique)  # Index du coup avec la plus haute probabilité
    coup_choisi = coups_possibles[index_coup_choisi]  # Choix du coup
    return coup_choisi

# Fonction pour jouer une partie générée automatiquement
def jouer_partie_auto_generee(reseau):
    plateau = chess.Board()  # Initialisation du plateau d'échecs
    historique_etats = []  # Liste pour stocker l'historique des états du plateau
    historique_politiques = []  # Liste pour stocker l'historique des politiques
    historique_coups = []  # Liste pour stocker l'historique des coups

    # Tant que la partie n'est pas terminée
    while not plateau.is_game_over():
        etat = transformer_etat(plateau)  # Transformation de l'état du plateau en un format utilisable par le réseau
        etat = np.expand_dims(etat, axis=0)  # Ajout d'une dimension supplémentaire pour correspondre à l'entrée du réseau
        politique = reseau.predict(etat)[0]  # Prédiction de la politique par le réseau
        coup = choisir_coup(plateau, politique)  # Choix du coup à jouer
        plateau.push(coup)  # Jouer le coup choisi

        # Ajout de l'état, de la politique et du coup à leurs historiques respectifs
        historique_etats.append(etat)
        historique_politiques.append(politique)
        historique_coups.append(coup_to_onehot(coup, plateau))

    # Détermination du résultat de la partie
    resultat = determiner_resultat(plateau)
    # Retour des historiques et du résultat
    return historique_etats, historique_politiques, historique_coups, resultat