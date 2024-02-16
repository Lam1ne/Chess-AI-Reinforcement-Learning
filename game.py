import chess
from MCTS import mcts

# Classe pour simuler une partie d'échecs
class ChessSimulator:
    # Initialisation de la classe
    def __init__(self, reseau):
        self.plateau = chess.Board()  # Création d'un nouveau plateau d'échecs
        self.reseau = reseau  # Réseau neuronal pour l'IA

    # Méthode pour faire jouer un coup à l'IA
    def jouer_coup_ia(self, itermax=100):
        # Utilisation de l'algorithme MCTS pour choisir le meilleur coup
        coup_choisi = mcts(self.plateau, itermax, self.reseau)
        # Jouer le coup choisi sur le plateau
        self.plateau.push(coup_choisi)

    # Méthode pour lancer une partie
    def run(self):
        # Tant que la partie n'est pas terminée
        while not self.plateau.is_game_over():
            # Afficher le plateau
            print(self.plateau)
            # Faire jouer un coup à l'IA
            self.jouer_coup_ia(itermax=100)

        # Afficher le résultat de la partie
        print("Partie terminée:", self.plateau.result())