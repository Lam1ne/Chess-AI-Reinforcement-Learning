import tensorflow as tf
from keras import layers, models
import yaml

class ChessNetwork:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
    # Méthode pour construire le modèle
    def build_model(self):
        # Chargement des paramètres à partir de la configuration
        input_shape = (8, 8, 112)
        num_filters = self.config['model']['filters']
        num_blocks = self.config['model']['residual_blocks']

        # Couche d'entrée
        inputs = layers.Input(shape=input_shape)

        # Couche de convolution initiale
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(inputs)

        # Blocs répétés avec convolution, squeeze-and-excitation, et connexions de saut
        for _ in range(num_blocks):
            x = self.build_residual_block(x, num_filters)

        # Tête de politique
        policy_head = layers.Conv2D(filters=2, kernel_size=1, activation='relu')(x)
        policy_head = layers.Flatten()(policy_head)
        policy_output = layers.Dense(73 * 8 * 8, activation='softmax', name='policy_output')(policy_head)

        # Tête de valeur
        value_head = layers.Conv2D(filters=1, kernel_size=1, activation='relu')(x)
        value_head = layers.Flatten()(value_head)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value_head)  # gagner, nul, perdre

        # Combinaison des têtes
        outputs = [policy_output, value_output]

        # Création du modèle
        model = models.Model(inputs=inputs, outputs=outputs)

        return model

    # Méthode pour construire un bloc résiduel
    def build_residual_block(self, x, num_filters):
        # Sauvegarde de la valeur d'entrée pour la connexion de saut
        block_input = x

        # Couches de convolution avec bloc squeeze-and-excitation entre les deux
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
        x = self.squeeze_and_excitation(x, num_filters)

        # Connexion de saut et activation ReLU
        x = layers.add([block_input, x])
        x = layers.ReLU()(x)

        return x

    # Méthode pour le squeeze-and-excitation
    def squeeze_and_excitation(self, input_tensor, num_filters, ratio=16):
        # Opération de squeeze
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Reshape((1, 1, num_filters))(se)
        se = layers.Dense(num_filters // ratio, activation='relu')(se)
        se = layers.Dense(num_filters, activation='sigmoid')(se)

        # Opération d'excitation
        x = layers.Multiply()([input_tensor, se])
        return x

    # Méthode pour compiler le modèle
    def compile_model(self):
        learning_rate = self.config['training']['lr_values'][0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                        loss={
                            'policy_output': 'categorical_crossentropy',
                            'value_output': 'mean_squared_error'
                        },
                        metrics={
                            'policy_output': 'accuracy',
                            'value_output': 'mean_squared_error'
                        })

    # Méthode pour entraîner le modèle
    def train(self, data, labels, verbose=1):
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['total_steps']
        return self.model.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # Méthode pour prédire l'état du plateau
    def predict(self, board_state):
        return self.model.predict(board_state)

    # Méthode pour sauvegarder le modèle
    def save(self, file_path):
        self.model.save(file_path)

    # Méthode pour charger le modèle
    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    # Méthode pour ajuster le taux d'apprentissage
    def adjust_learning_rate(self):
        new_learning_rate = self.config['training']['lr_values'][0]
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate),
                        loss={
                            'policy_output': 'categorical_crossentropy',
                            'value_output': 'mean_squared_error'
                        },
                        metrics={
                            'policy_output': 'accuracy',
                            'value_output': 'mean_squared_error'
                        })