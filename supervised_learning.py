import numpy as np
from utils.chess_utils import load_pgn, coup_to_onehot
from network import ChessNetwork
from config import config

def split_data(data, test_ratio=0.2):
    """Diviser les données en ensembles d'entraînement et de test."""
    np.random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

def prepare_data(data):
    """Préparer les données pour l'entraînement."""
    X, y = zip(*data)
    return np.array(X), np.array(y)


def main():
    pgn_file_path = 'opening.pgn'
    # Charger et préparer les données   
    data = load_pgn(pgn_file_path)
    train_data, test_data = split_data(data)
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Créer et entraîner le réseau
    chess_network = ChessNetwork(config)
    chess_network.compile_model()
    chess_network.train(X_train, y_train, verbose = 1)  # Modification ici

    # Évaluer le modèle
    evaluation_results = chess_network.model.evaluate(X_test, y_test)
    print(f"Total Loss: {evaluation_results[0]}")
    print(f"Policy Output Loss: {evaluation_results[1]}")
    print(f"Value Output Loss: {evaluation_results[2]}")
    print(f"Policy Output Accuracy: {evaluation_results[3]}")
    print(f"Value Output MSE: {evaluation_results[4]}")


    # Sauvegarder le modèle
    chess_network.save('chess_model.h5')

if __name__ == '__main__':
    main()
