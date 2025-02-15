from flask import Flask, request, render_template
import os
import json
from web3 import Web3
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Blockchain Configuration
WEB3_PROVIDER = 'http://127.0.0.1:8545'

try:
    w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
    if not w3.is_connected():
        raise ConnectionError("Web3 provider is not running. Start Ganache or your blockchain node.")

    # Load blockchain contract
    contract_path = 'C:/Users/thanm/Downloads/CombinedDataset/CombinedDataset/blockchain/build/contracts/TBRecords.json'
    
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"❌ Blockchain contract file not found: {contract_path}")

    with open(contract_path, 'r') as f:
        contract_data = json.load(f)

    network_id = list(contract_data['networks'].keys())[0]  # Automatically select available network
    contract_address = contract_data['networks'][network_id]['address']
    contract = w3.eth.contract(address=contract_address, abi=contract_data['abi'])

except Exception as e:
    print(f"❌ Error loading blockchain contract: {e}")
    contract = None  # Disable blockchain if it fails

# ✅ Load Full Model
MODEL_H5_PATH = "full_model.h5"

if not os.path.exists(MODEL_H5_PATH):
    print(f"❌ ERROR: Model file '{MODEL_H5_PATH}' not found. Run 'recover_model.py' first!")
    model = None
else:
    try:
        model = tf.keras.models.load_model(MODEL_H5_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# ✅ Class Labels
CLASS_LABELS = [
    'PTB', 'normal', 'ATB', 'tuberculosis pleuritis',
    'NATB', 'STB', '', 'TB', 'right upper pneumonia'
]

# ✅ Impact Counter
impact_counter = 1000

# ✅ Image Processing Function
def process_image(file_path):
    """Resize and preprocess image for model input."""
    img = cv2.imread(file_path)
    img = cv2.resize(img, (100, 100))
    return img.reshape(-1, 100, 100, 3)

# ✅ Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global impact_counter

    if model is None:
        return "❌ Error: Model not loaded. Run 'recover_model.py' first.", 500

    file = request.files['file']
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Get Prediction
    processed_img = process_image(file_path)
    prediction = model.predict(processed_img)
    result = CLASS_LABELS[np.argmax(prediction)]

    # Blockchain Transaction
    tx_hash = None
    if contract:
        try:
            account = w3.eth.accounts[0]
            tx_hash = contract.functions.addRecord(
                filename,
                result
            ).transact({'from': account, 'gas': 500000})
            tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            block_number = tx_receipt['blockNumber']
        except Exception as e:
            print(f"❌ Blockchain error: {e}")
            block_number = "N/A"
    else:
        block_number = "N/A"

    # Update Impact Counter
    impact_counter += 1

    return render_template(
        'result.html',
        filename=filename,
        prediction=result,
        tx_hash=tx_hash.hex() if tx_hash else "Blockchain disabled",
        block_number=block_number,
        impact_count=impact_counter
    )

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5000)
