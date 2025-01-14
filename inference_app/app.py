from inference import Inference
from flask import Flask, request, jsonify

app = Flask(__name__)

# inference = Inference(s3_url='s3://model_martifacts/parspec_interview/trained_model.pkl')
inference = Inference(model_path='trained_model.pkl')

@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'})
    try:
        prediction, confidence = inference.predict(url)
        return jsonify({'predicted_label': prediction, 'confidence': confidence})
    except:
        return jsonify({'error': 'Failed to extract text from the URL'})

if __name__ == "__main__":
    app.run(debug=True)