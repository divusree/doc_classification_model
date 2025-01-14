import joblib
import requests
from pypdf import PdfReader
import io
import boto3
import argparse


class Inference:
    def __init__(self, model_path=None, s3_url=None):
        if model_path:
            self.model = joblib.load(model_path)
        elif s3_url:
            self.model = self.load_model_from_s3(s3_url)
        else:
            raise ValueError("Either model_path or s3_url must be provided")
        self.vectorizer = self.model.named_steps["vectoriser"]
        self.network = self.model.named_steps["logisticreg"]

    def load_model_from_s3(self, s3_url):
        s3 = boto3.client("s3")
        bucket_name, key = self.parse_s3_url(s3_url)
        with io.BytesIO() as f:
            s3.download_fileobj(bucket_name, key, f)
            f.seek(0)
            model = joblib.load(f)
        return model

    def parse_s3_url(self, s3_url):
        parts = s3_url.replace("s3://", "").split("/")
        bucket_name = parts[0]
        key = "/".join(parts[1:])
        return bucket_name, key

    def load_data_from_url(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
        }
        try:
            r = requests.get(
                url, stream=True, headers=headers, allow_redirects=True, timeout=5
            )
            reader = PdfReader(io.BytesIO(r.content))
            text = reader.pages[0].extract_text()
            return text, len(reader.pages)
        except Exception as e:
            print(f"Exception Raised for {url} -- {e}")
            return "", 0

    def predict(self, url):
        text, num_pages = self.load_data_from_url(url)
        if num_pages > 0:
            text_transformed = self.vectorizer.transform([text])
            prediction = self.network.predict(text_transformed)
            confidence = float(self.network.predict_proba(text_transformed).max())
            return prediction[0], confidence
        else:
            return "", 0


def main():
    parser = argparse.ArgumentParser(
        description="Inference Run for Document Classification model."
    )
    parser.add_argument("--model_path", type=str, help="Path to saved model")
    parser.add_argument("--url", type=str, help="url")
    args = parser.parse_args()

    inference = Inference(model_path=args.model_path)
    prediction, confidence = inference.predict(args.url)
    print({"predicted_label": prediction, "confidence": confidence})


if __name__ == "__main__":
    main()
