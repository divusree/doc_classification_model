import requests
import io
import json
from collections import defaultdict
from pypdf import PdfReader
import pandas as pd

class DataLoader:
    def __init__(self, headers=None):
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36',
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br"
            }
        self.headers = headers

    def load_data_from_url(self, url):
        """
        Load data from a URL and return the extracted text and number of pages.
        """
        try:
            response = requests.get(url, stream=True, headers=self.headers, allow_redirects=True, timeout=5)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
            text = {str(idx): page.extract_text() for idx, page in enumerate(reader.pages[:4])}
            num_pages = len(reader.pages)
            return text, num_pages
        except requests.RequestException as e:
            print(f"Request exception for {url}: {e}")
        except Exception as e:
            print(f"Exception raised for {url}: {e}")
        return None, 0

    def create_dataset_json(self, df, save_path="dataset.json"):
        """
        Create a dataset JSON from a DataFrame and save it to a file.
        """
        err_count = 0
        dataset = defaultdict(dict)

        for index, row in df[df['valid_links'] == True].iterrows():
            url, label = row['datasheet_link'], row['target_col']
            text, num_pages = self.load_data_from_url(url)
            if text:
                dataset[url] = {
                    'text': text,
                    'num_pages': num_pages,
                    'label': label
                }
                df.at[index, 'valid_links'] = True
                print(f"At {index}, {url} has {num_pages} number of pages.")
            else:
                df.at[index, 'valid_links'] = False
                err_count += 1
                print(f'-----------{index}--------------')

            if index % 50 == 0:
                self.save_dataset(dataset, save_path, index, err_count)

        self.save_dataset(dataset, save_path, len(df), err_count)
        return dataset

    def save_dataset(dataset, save_path, index, err_count):
        """
        Save the dataset to a JSON file.
        """
        with open(save_path, 'w') as f:
            json.dump(dataset, f)
        print(f"Finished {index}. Error count: {err_count}")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("dataset.xlsx", sheet_name = None)  
    data_loader = DataLoader()
    data_loader.create_dataset_json(df)