from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import argparse
import joblib
import os
import json
class ModelInference:
    def __init__(self, data_json, model_path = "trained_model.pkl"):
        """
        Initialize the ModelTrainer with text, associated labels and the trained model.
        """
        self.data_json = data_json
        self.urls_list, self.targets, self.corpus = self.prepare_data(data_json)
        self.clf = joblib.load(model_path)
        self.vectorizer = self.clf.named_steps["vectoriser"]
        self.network = self.clf.named_steps["logisticreg"]
        self.labels = ['lighting', 'fuses', 'others', 'cable']

    def prepare_data(self, data_json):
        """
        Prepare data from the JSON structure.
        """
        urls_list = list(data_json.keys())
        labels = [data_json[url].get('label') for url in urls_list if data_json[url].get('label') is not None]
        corpus = [data_json[url]['text'].get('0') for url in urls_list if data_json[url]['text'].get('0') is not None]
        return urls_list, labels, corpus

    def print_misclassified_samples(self, y_test, preds, confidence_list):
        """
        Print misclassified samples.
        """
        print("The following are misclassified examples ----")
        for idx, p in enumerate(y_test):
            if p != preds[idx]:
                print("true values", p, "predictions", preds[idx], "confidence", float(confidence_list[idx].max()), self.urls_list[idx])
        print (classification_report(y_test, preds) ) 
    def visualiser(self, y_test, preds):
        """
        Visualize the confusion matrix.
        """
        if not os.path.exists('metrics'):
            os.makedirs('metrics')

        cm = confusion_matrix(y_test, preds, labels=self.labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.labels)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix for test set")
        plt.savefig('metrics/confusion_matrix_test.png')
        plt.show()

    def predict_all(self, visualise=True):
        """
        Run predictions for all datapoints.
        """
        text_transformed = self.vectorizer.transform(self.corpus)
        prediction = self.network.predict(text_transformed)
        confidence_list = self.network.predict_proba(text_transformed)
        self.print_misclassified_samples(self.targets, prediction, confidence_list)
        if visualise:
            self.visualiser(self.targets, prediction)

    def predict(self, text):
        """
        Run prediction for just 1 datapoint.
        """
        text_transformed = self.vectorizer.transform([text])
        prediction = self.network.predict(text_transformed)
        confidence = float(self.network.predict_proba(text_transformed).max())
        return prediction, confidence    
      
def main():
    parser = argparse.ArgumentParser(description='Run inference on a document classification model.')
    parser.add_argument('--test_json', type=str, help='Path to the test JSON file')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    args = parser.parse_args()

    try:
        with open(args.test_json, 'r') as f:
            test_json = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    trainer = ModelInference(test_json, model_path = args.model_path)
    trainer.predict_all(visualise=True)
if __name__ == "__main__":
    main()