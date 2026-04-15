import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow import random

random.set_seed(666) # set tf random seed to ensure reproducibility

# Load data
train_csvs = sorted([_.path for _ in os.scandir("data") if "train" in _.name])
test_csvs = sorted([_.path for _ in os.scandir("data") if "test" in _.name])
train_dfs = [pd.read_csv(_).rename(columns={'# x':"x", ' y':"y"}) for _ in train_csvs]
test_dfs = [pd.read_csv(_).rename(columns={'# x':"x", ' y':"y"}) for _ in test_csvs]

# Build one hidden layer model with variable n_hidden_neurons and learning rate(lr)
def build_model(n_hidden_neurons, lr):

    model = models.Sequential([
        layers.Dense(n_hidden_neurons, activation='tanh', input_shape=(1,)),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='mse'
    )
    
    return model



# assign some fixed quantities(hyperparameters) for training
min_lr=1e-6
max_lr=1e-1
lr_steps=30
epochs = 2000
batch_size = 32
validation_split = 0.15
patience = 10 # Number of epochs with no improvement after which training will be stopped

# start tuning lr
def tune_lr(x_train, y_train, n_hidden_neurons):
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
        # chose min loss within all epochs
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_lr = lr
    
    return best_lr


# Ensembles
def ensemble_predictor(x_train, y_train, x_test, n_hidden_neurons, ensemble_size):
    # tune LR for every different input
    best_lr = tune_lr(x_train, y_train, n_hidden_neurons)
    y_preds = [] # collect every ensemble prediction
    
    for _ in range(ensemble_size):
        model = build_model(n_hidden_neurons, best_lr)
        
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )
        
        y_pred = model.predict(x_test, verbose=0) # shape = (#(x_test), 1)
        y_preds.append(y_pred)
    
    y_preds = np.array(y_preds) # shape = (#(ensembles), #(x_test), 1)
    
    return best_lr, y_preds


n_hiddens = [2,3,4,5,6,7,8]
n_ensembles = [25, 50, 100]


best_lr_results = []

for train_df, test_df, train_csv in zip(train_dfs, test_dfs, train_csvs):
    sigma = os.path.basename(train_csv).split("_")[1]
    x_train = train_df["x"]
    y_train = train_df["y"]
    x_test = test_df["x"]
    y_test = test_df["y"]

    for n_hidden in n_hiddens:
        for n_ensemble in n_ensembles:
            best_lr, y_preds = ensemble_predictor(x_train, y_train, x_test, n_hidden, n_ensemble)
            y_preds_output_path = os.path.join("preds", f"{sigma}_{n_hidden}_{n_ensemble}.npy")
            np.save(y_preds_output_path, y_preds) # since training takes lots of time, keep evaluation step in another script
            best_lr_results.append([sigma, n_hidden, n_ensemble, best_lr])
            print(f"{sigma}: hidden={n_hidden}, ensemble={n_ensemble}, best_lr={best_lr}")

best_lr_results_df = pd.DataFrame(best_lr_results, columns=["sigma", "n_hidden", "n_ensemble", "best_lr"])
best_lr_results_df.to_csv("best_lr_results.csv", index=False)
