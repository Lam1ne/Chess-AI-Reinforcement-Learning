import chess
import numpy as np
from MCTS import mcts
from utils.chess_utils import determiner_resultat, transformer_etat, enregistrer_partie_en_pgn, calculate_policy_label, calculate_value_label
import random 


# Classe pour l'auto-apprentissage
class AutoApprentissageLC0:
    def __init__(self, reseau, nombre_parties=100, itermax=100, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995): # Ajouter les paramètres nécessaires
        self.reseau = reseau # Le réseau à entraîner
        self.nombre_parties = nombre_parties # Le nombre de parties à jouer
        self.itermax = itermax # Le nombre d'itérations pour le MCTS
        self.historique = [] # L'historique des parties jouées
        self.compteur_parties = 0 # Le compteur de parties
        self.epsilon = epsilon_start # Le paramètre epsilon pour epsilon-greedy
        self.epsilon_end = epsilon_end # La valeur finale de epsilon
        self.epsilon_decay = epsilon_decay # Le taux de dégradation de epsilon

    # Méthode pour jouer une partie
    def jouer_une_partie(self):
        plateau = chess.Board()  # Initialise un nouveau plateau de jeu
        try:
            # Continue à jouer tant que la partie n'est pas terminée
            while not plateau.is_game_over():
                etat = transformer_etat(plateau)  # Transforme l'état du plateau en un format utilisable par le réseau
                # Implémente la stratégie epsilon-greedy
                if random.uniform(0, 1) < self.epsilon:
                    coup = random.choice([move for move in plateau.legal_moves])  # Choisis un coup aléatoire
                else:
                    coup = mcts(plateau, self.itermax, self.reseau)  # Utilise le MCTS pour choisir un coup
                if coup is None:
                    break
                plateau.push(coup)  # Joue le coup choisi
                print(coup)
                print(plateau)
                self.historique.append((etat, coup, None))  # Ajoute l'état et le coup à l'historique
                # Dégrade epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        except Exception as e:
            print("Une erreur est survenue:", e)
            raise e
        # Met à jour les labels de politique et de valeur pour l'entraînement
        game_result = determiner_resultat(plateau)
        for i, (state, move, _) in enumerate(self.historique): 
            policy_label = calculate_policy_label(plateau, move) 
            value_label = calculate_value_label(game_result)
            self.historique[i] = (state, policy_label, value_label)

            # Log le résultat de la partie
            print(f"Résultat de la partie: {plateau.result()}")

            # Incrémente le compteur de parties et sauvegarde la partie
            self.compteur_parties += 1
            enregistrer_partie_en_pgn(plateau, f"partie_{self.compteur_parties}.pgn")

            return determiner_resultat(plateau)

    # Prépare les données pour l'entraînement du réseau
    def preparer_donnees(self):
        X = np.array([etat for etat, _, _ in self.historique])  # Les états du jeu
        y_policy = np.array([politique for _, politique, _ in self.historique])  # Les politiques (coups joués)
        y_value = np.array([resultat for _, _, resultat in self.historique])  # Les résultats des parties
        return X, y_policy, y_value

    # Entraîne le réseau
    def entrainer_reseau(self):
        X, y_policy, y_value = self.preparer_donnees()  # Prépare les données
        self.reseau.train(X, {'policy_output': y_policy, 'value_output': y_value})  # Entraîne le réseau

    # Lance l'auto-apprentissage
    def lancer_auto_apprentissage(self):
        for i in range(self.nombre_parties):  # Joue le nombre de parties spécifié
            resultat = self.jouer_une_partie()  # Joue une partie
            if (i + 1) % 1 == 0:  # Sauvegarde le modèle toutes les 10 parties
                self.reseau.save(f'chemin_du_modele_apres_{i + 1}_parties.h5')

        # Entraîne le réseau
        self.entrainer_reseau()
        # Sauvegarde le modèle final
        self.reseau.save('Modele_finale.h5')