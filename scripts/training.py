import os
import joblib
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

class ModelTrainer:
    def __init__(self, labels, data_json = 'datasets/train_json.json'):
        """
        Initialize the ModelTrainer with labels and data.
        """
        self.labels = labels
        self.data_json = data_json
        self.encoding = {label: idx for idx, label in enumerate(labels)}
        self.urls_list, self.train_labels, self.train_corpus = self.prepare_data(data_json)
        self.train_df = pd.DataFrame({"text": self.train_corpus, 'labels': self.train_labels})
        self.clf = None

    def prepare_data(self, data_json):
        """
        Prepare data from the JSON structure.
        """
        urls_list = list(data_json.keys())
        labels = [data_json[url].get('label') for url in urls_list if data_json[url].get('label') is not None]
        corpus = [data_json[url]['text'].get('0') for url in urls_list if data_json[url]['text'].get('0') is not None]
        return urls_list, labels, corpus

    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        return train_test_split(
            self.train_df['text'], self.train_df['labels'], test_size=0.2, random_state=0, shuffle=True, stratify=self.train_df['labels']
        )

    def create_and_train_model(self, X_train, y_train):
        """
        Create and train the model.
        """
        clf = Pipeline([
            ('vectoriser', TfidfVectorizer(strip_accents='ascii', stop_words='english', norm='l2')),
            ('logisticreg', LogisticRegression(solver='sag', class_weight='balanced'))
        ])
        clf.fit(X_train, y_train)
        return clf

    def evaluate_model(self, clf, X_test, y_test):
        """
        Evaluate the model and return predictions.
        """
        score = clf.score(X_test, y_test)
        print(f"Model accuracy: {score}")
        return clf.predict(X_test), clf.predict_proba(X_test)

    def print_misclassified_samples(self, y_test, preds, confidence_list):
        """
        Print misclassified samples.
        """
        print("The following are misclassified examples ----")
        for idx, (ridx, p) in enumerate(y_test.items()):
            if p != preds[idx]:
                print("true values", p, "predictions", preds[idx], "confidence", float(confidence_list[idx].max()), self.urls_list[ridx])

    def visualiser(self, y_test, preds):
        """
        Visualize the confusion matrix and PCA plot.
        """
        if not os.path.exists('metrics'):
            os.makedirs('metrics')

        cm = confusion_matrix(y_test, preds, labels=self.labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.labels)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix for training set")
        plt.savefig('metrics/confusion_matrix_train.png')

        vectorizer = TfidfVectorizer()
        train_transform = vectorizer.fit_transform(self.train_corpus)
        reduced_data = PCA(n_components=2).fit_transform(train_transform)

        labels_color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}

        fig, ax = plt.subplots()
        for index, instance in enumerate(reduced_data):
            pca_comp_1, pca_comp_2 = instance
            color = labels_color_map[self.encoding[self.train_labels[index]]]
            ax.scatter(pca_comp_1, pca_comp_2, c=color)
        plt.title("PCA plot for training set")
        plt.savefig('metrics/pca_plot_train.png')
        plt.show()

    def train_and_evaluate(self, visualise=True):
        """
        Train and evaluate the model, optionally visualizing the results.
        """
        X_train, X_test, y_train, y_test = self.split_data()
        self.clf = self.create_and_train_model(X_train, y_train)
        preds, confidences = self.evaluate_model(self.clf, X_test, y_test)
        self.print_misclassified_samples(y_test, preds, confidences)
        if visualise:
            self.visualiser(y_test, preds)

    def save_model(self, save_path= 'trained_model.pkl'):
        """
        Save the trained model.
        """
        if not os.path.exists('models'):
            os.makedirs('models')        
        joblib.dump(self.clf, os.path.join('models', save_path))

def main():
    parser = argparse.ArgumentParser(description='Train a text classification model.')
    parser.add_argument('--train_json', type=str, help='Path to the training JSON file')
    args = parser.parse_args()

    try:
        with open(args.train_json, 'r') as f:
            train_json = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    labels = ['lighting', 'fuses', 'others', 'cable']

    trainer = ModelTrainer(labels, train_json)
    trainer.train_and_evaluate(visualise=True)
    trainer.save_model()
    
if __name__ == "__main__":
    main()