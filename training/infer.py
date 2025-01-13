from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
        self.urls_list, self.labels, self.corpus = self.prepare_data(data_json)
        self.clf = joblib.load(model_path)
        self.vectorizer = self.clf.named_steps["vectoriser"]
        self.network = self.clf.named_steps["logisticreg"]

    def prepare_data(self, data_json):
        """
        Prepare data from the JSON structure.
        """
        urls_list = list(data_json.keys())
        labels = [data_json[url].get('label') for url in urls_list if data_json[url].get('label') is not None]
        corpus = [data_json[url]['text'].get('0') for url in urls_list if data_json[url]['text'].get('0') is not None]
        return urls_list, labels, corpus

    def print_misclassified_samples(self, y_test, preds):
        """
        Print misclassified samples.
        """
        print("The following are misclassified examples ----")
        for idx, p in enumerate(y_test):
            if p != preds[idx]:
                print("true", p, "preds", preds[idx], self.urls_list[idx])

    def visualiser(self, y_test, preds):
        """
        Visualize the confusion matrix.
        """
        if not os.path.exists('metrics'):
            os.makedirs('metrics')

        cm = confusion_matrix(y_test, preds, labels=self.labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.labels)
        disp.plot()
        plt.savefig('metrics/confusion_matrix_test.png')

    def predict_all(self, visualise=True):
        text_transformed = self.vectorizer.transform(self.corpus)
        prediction = self.network.predict(text_transformed)
        self.print_misclassified_samples(self.labels, prediction)
        if visualise:
            self.visualiser(self.labels, prediction)

    def predict(self, text):
        text_transformed = self.vectorizer.transform([text])
        prediction = self.network.predict(text_transformed)
        return prediction      
      
def main():
    parser = argparse.ArgumentParser(description='Run inference on a document classification model.')
    parser.add_argument('--test_json', type=str, help='Path to the test JSON file')
    args = parser.parse_args()

    try:
        with open(args.train_json, 'r') as f:
            test_json = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    labels = ['lighting', 'fuses', 'others', 'cable']

    trainer = ModelInference(labels, test_json)
    trainer.predict_all(visualise=True)
if __name__ == "__main__":
    main()