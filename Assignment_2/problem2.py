import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers, models, callbacks, optimizers

np.random.seed(666)

# Load data
train_csvs = sorted([_.path for _ in os.scandir("data") if "train" in _.name])
test_csvs = sorted([_.path for _ in os.scandir("data") if "test" in _.name])
train_dfs = [pd.read_csv(_).rename(columns={'# x':"x", ' y':"y"}) for _ in train_csvs]
test_dfs = [pd.read_csv(_).rename(columns={'# x':"x", ' y':"y"}) for _ in test_csvs]

# Build one hidden layer model
def build_model(n_hidden_neurons, lr):
    # lr = learning rate
    model = models.Sequential([
        layers.Dense(n_hidden_neurons, activation='tanh', input_shape=(1,)),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='mse'
    )
    
    return model

# Tune lr

def tune_lr(x_train, y_train, n_hidden_neurons):
    # assign some fixed quantities
    min_lr=1e-6
    max_lr=1e-1
    lr_steps=30
    epochs = 100
    batch_size = 50
    validation_split = 0.15
    patience = 5 # Number of epochs with no improvement after which training will be stopped.
    # start tuning
    lr_candidates = np.logspace(np.log10(min_lr), np.log10(max_lr), lr_steps)
    best_lr = None
    best_loss = np.inf
    
    for lr in lr_candidates:
        model = build_model(n_hidden_neurons, lr)
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=patience, restore_best_weights=True)
            ]
        )
        # chose min loss within all epochs(we all know that epochs /, val loss \)
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_lr = lr
    
    return best_lr


# Ensembles
def train_ensemble(x_train, y_train, x_test, n_hidden_neurons, ensemble_size):
    
    # tune LR for every different input
    best_lr = tune_lr(x_train, y_train, n_hidden_neurons)
    
    y_preds = []
    
    for _ in range(ensemble_size):
        model = build_model(n_hidden_neurons, best_lr)
        
        model.fit(
            x_train, y_train,
            epochs=200,
            batch_size=16,
            validation_split=0.15,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True)
            ]
        )
        
        y_pred = model.predict(x_test, verbose=0) # shape = (#(x_test), 1)
        y_preds.append(y_pred)
    
    y_preds = np.array(y_preds) # shape = (#(ensembles), #(x_test), 1)
    
    return y_preds


for train_df, test_df in zip(train_dfs, test_dfs):
    x_train = train_df["x"]
    y_train = train_df["y"]
    x_test = test_df["x"]
    y_test = test_df["y"]
    train_ensemble(x_train, y_train, x_test, 5, 10)
    break