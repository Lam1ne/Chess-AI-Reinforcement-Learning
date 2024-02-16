import math
import random
import numpy as np
import chess
from utils.chess_utils import transformer_etat, coups_index, determiner_resultat
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class NoeudMCTS:
    def __init__(self, plateau, coup, parent):
        # Initialisation d'un noeud dans l'arbre MCTS
        self.plateau = plateau  # L'état actuel du plateau d'échecs
        self.coup = coup  # Le coup qui a conduit à cet état du plateau
        self.parent = parent  # Le noeud parent dans l'arbre MCTS
        self.enfants = []  # Les noeuds enfants dans l'arbre MCTS
        self.victoires = 0  # Le nombre de victoires simulées à partir de cet état
        self.visites = 0  # Le nombre de fois que ce noeud a été visité
        self.coups_non_essayes = list(plateau.legal_moves)  # Les coups qui n'ont pas encore été essayés à partir de cet état
        self.etat = str(plateau)  # L'état du plateau sous forme de chaîne de caractères

    def uct_selection_enfant(self):
        # Sélection d'un enfant pour l'expansion en utilisant l'UCT (Upper Confidence Bound 1 applied to Trees)
        uct_k = math.sqrt(2)  # Constante pour l'UCT
        parent_visites = self.parent.visites if self.parent is not None else 1  # Le nombre de visites du noeud parent

        def uct_value(c):
            # Calcul de la valeur UCT pour un noeud enfant
            victoires_scalar = np.sum(c.victoires)  # Le nombre total de victoires pour ce noeud
            # Calcul de la valeur UCT
            return victoires_scalar / c.visites + uct_k * math.sqrt(2 * math.log(parent_visites) / (c.visites if c.visites > 0 else 1)) + random.uniform(0, 0.01)

        # Sélection de l'enfant avec la plus grande valeur UCT
        selectionne = max(self.enfants, key=uct_value)
        return selectionne

    def ajouter_enfant(self, m, b):
        # Ajout d'un nouvel enfant à la liste des enfants
        nouveau_noeud = NoeudMCTS(plateau=b, coup=m, parent=self)  # Création du nouveau noeud
        self.coups_non_essayes.remove(m)  # Suppression du coup de la liste des coups non essayés
        self.enfants.append(nouveau_noeud)  # Ajout du nouveau noeud à la liste des enfants
        return nouveau_noeud

    def mettre_a_jour(self, resultat):
        # Mise à jour des statistiques du noeud après la simulation
        self.visites += 1  # Incrémentation du nombre de visites
        self.victoires += resultat  # Ajout du résultat à la somme des victoires

    def selectionner_coup_avec_reseau(self, reseau):
        # Sélection d'un coup en utilisant le réseau neuronal
        etat = transformer_etat(self.plateau)  # Transformation de l'état du plateau pour le réseau
        etat = np.expand_dims(etat, axis=0)  # Ajout d'une dimension pour la prédiction

        politique, _ = reseau.predict(etat)  # Prédiction de la politique par le réseau
        politique = politique.flatten()  # Aplatissage de la politique

        # Création d'une liste de coups possibles et de leurs scores
        coups_possibles = list(self.plateau.legal_moves)
        coups_scores = [(coup, politique[coups_index(coup, self.plateau)]) for coup in coups_possibles]
        coups_scores.sort(key=lambda x: x[1], reverse=True)  # Tri des coups par score

        # Sélection du coup avec le score le plus élevé qui n'a pas encore été essayé
        for coup, score in coups_scores:
            if coup in self.coups_non_essayes:
                return coup

        # Si aucun coup n'a été trouvé, sélection d'un coup au hasard parmi les coups non essayés
        if self.coups_non_essayes:
            selected_random_move = random.choice(self.coups_non_essayes)
            return selected_random_move

    def simuler(self):
        # Simulation d'une partie à partir de l'état actuel jusqu'à la fin
        temp_board = self.plateau.copy()  # Copie du plateau actuel
        while not temp_board.is_game_over():  # Tant que la partie n'est pas terminée
            coup = random.choice(list(temp_board.legal_moves))  # Sélection d'un coup aléatoire
            temp_board.push(coup)  # Application du coup sur le plateau
        return determiner_resultat(temp_board)  # Retour du résultat de la partie

    def retropropagation(self, resultat, reseau):
        # Rétropropagation du résultat de la simulation vers les noeuds parents
        etat = transformer_etat(self.plateau)  # Transformation de l'état du plateau pour le réseau
        etat = np.expand_dims(etat, axis=0)  # Ajout d'une dimension pour la prédiction
        _, valeur = reseau.predict(etat)  # Prédiction de la valeur par le réseau

        self.mettre_a_jour(valeur)  # Mise à jour des statistiques du noeud
        if self.parent:  # Si le noeud a un parent
            self.parent.retropropagation(resultat, reseau)  # Rétropropagation du résultat au noeud parent

def mcts(etat_racine, itermax, reseau):
    # Implémentation de l'algorithme MCTS
    transposition_table = {}  # Table de transposition pour stocker les noeuds visités
    noeud_racine = NoeudMCTS(plateau=etat_racine)  # Création du noeud racine
    transposition_table[noeud_racine.etat] = noeud_racine  # Stockage du noeud racine dans la table de transposition

    for iteration in range(itermax):  # Pour chaque itération
        noeud = noeud_racine  # On commence par le noeud racine
        etat = etat_racine.copy()  # On copie l'état racine

        # Sélection avec réseau
        while noeud.coups_non_essayes == [] and noeud.enfants != []:  # Tant qu'il n'y a pas de coups non essayés et qu'il y a des enfants
            if noeud.plateau.is_game_over():  # Si la partie est terminée
                break  # On sort de la boucle
            noeud = noeud.uct_selection_enfant()  # On sélectionne un enfant avec l'UCT
            etat.push(noeud.coup)  # On applique le coup de l'enfant sur l'état

        # Expansion avec réseau
        if noeud.coups_non_essayes != []:  # S'il y a des coups non essayés
            coup = noeud.selectionner_coup_avec_reseau(reseau)  # On sélectionne un coup avec le réseau
            etat.push(coup)  # On applique le coup sur l'état
            if str(etat) in transposition_table:  # Si l'état est déjà dans la table de transposition
                noeud = transposition_table[str(etat)]  # On récupère le noeud correspondant
            else:  # Sinon
                noeud = noeud.ajouter_enfant(coup, etat)  # On ajoute un nouvel enfant avec le coup et l'état
                transposition_table[str(etat)] = noeud  # On stocke le nouveau noeud dans la table de transposition

        # Simulation
        resultat = noeud.simuler()  # On simule une partie à partir de l'état actuel

        # Rétropropagation
        noeud.retropropagation(resultat, reseau)  # On rétropropage le résultat de la simulation

    if noeud_racine.enfants:  # Si le noeud racine a des enfants
        # On retourne le coup de l'enfant avec le plus grand ratio victoires/visites
        return max(noeud_racine.enfants, key=lambda c: np.sum(c.victoires) / c.visites).coup
    else:  # Sinon
        return None  # On retourne None