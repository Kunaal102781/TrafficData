from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and label encoders
with open(r"C:\Users\kunal\OneDrive\Desktop\DCN\WebInterface\model_and_encoders.pkl", 'rb') as f:
    loaded = pickle.load(f)
    model = loaded['model']
    label_encoders = loaded['label_encoders']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    input_data = request.json

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the same preprocessing (label encoding) to the input data
    for col in label_encoders.keys():
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col].astype(str))
        else:
            input_df[col] = input_df[col].astype(str)
    
    # Ensure the input data has the same feature columns as the training data
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    return jsonify({'sniffed': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
