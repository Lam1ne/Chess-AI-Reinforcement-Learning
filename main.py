from self_learning import AutoApprentissageLC0
import tensorflow as tf
from network import ChessNetwork
from config import config
def main():
    reseau = ChessNetwork(config)
    auto_apprentissage = AutoApprentissageLC0(reseau, nombre_parties=100, itermax=100, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.99)
    auto_apprentissage.lancer_auto_apprentissage()

if __name__ == "__main__":
    main()