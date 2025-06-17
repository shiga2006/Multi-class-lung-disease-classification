from flask import Flask, request, jsonify, render_template
import os
from inference import classify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = "temp_upload.jpg"
    try:
        print("Saving uploaded file...")
        file.save(filename)
        print("Calling classify...")
        result = classify(filename)
        print("Classify result:", result)
        os.remove(filename)
        print("Returning prediction")
        return jsonify({
            'prediction': result,
            'classes': [
                "viral pneumonia",
                "bacterial pneumonia",
                "normal",
                "covid",
                "tuberculosis"
            ]
        })
    except Exception as e:
        print("Exception during prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)