config = {
    'model': {
        'filters': 256,  
        'residual_blocks': 19 
    },
    'training': {
        'batch_size': 30000,  
        'total_steps': 20,  # Nombre de pas/epoch
        'lr_values': [
            0.0001  
        ]
    }
}
