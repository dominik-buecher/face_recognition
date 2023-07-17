from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Define a function that creates your model
def create_model(optimizer='rmsprop', dropout_rate=0.5):
    InceptionV3_base_model = InceptionV3(include_top=False, input_shape=input_shape, weights=weights)
    
    for layer in InceptionV3_base_model.layers:
        layer.trainable = False
    
    InceptionV3_model = tf.keras.Sequential([
        InceptionV3_base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    InceptionV3_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return InceptionV3_model

# Create a KerasClassifier wrapper for scikit-learn compatibility
model = KerasClassifier(build_fn=create_model)

# Define the hyperparameters and their values to search
param_grid = {
    'epochs': [50, 100, 150],  # Example values for epochs
    'steps_per_epoch': [10, 20, 30]  # Example values for steps_per_epoch
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(train_generator)

# Get the best hyperparameters and the corresponding model
best_params = grid_result.best_params_
best_model = grid_result.best_estimator_.model
