import h5py
import json
import tensorflow as tf

model_path = "Model/keras_model.h5"

# Load the model configuration
with h5py.File(model_path, 'r') as f:
    model_config = f.attrs.get('model_config')

# Decode the JSON configuration
model_config = json.loads(model_config.decode('utf-8'))

# Recursively remove the 'groups' argument from the configuration
def remove_groups(config):
    if isinstance(config, dict):
        if 'groups' in config:
            del config['groups']
        for key, value in config.items():
            remove_groups(value)
    elif isinstance(config, list):
        for item in config:
            remove_groups(item)

remove_groups(model_config)

# Save the updated configuration back to the file
with h5py.File(model_path, 'a') as f:
    f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

# Try loading the model again
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
