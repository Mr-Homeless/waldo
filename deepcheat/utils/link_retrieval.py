import argparse
import os
import pandas as pd
import requests
import validators
from common import ensure_dir_exists
from pathlib import Path

# Set up command line arguments
parser = argparse.ArgumentParser(description="Get URL's from API and store them locally")
parser.add_argument("-e", "--endpoint", help='Target URL on server', required=False,
                    default="https://waldo.vision/api/analysis/urls", type=str)
parser.add_argument("-k", "--key", help='API Key', required=False,
                    default=os.environ.get("WALDO_API_KEY"), type=str)
parser.add_argument("-i", "--id", help='API Key ID', required=False, type=str,
                    default=os.environ.get("WALDO_API_ID"))
parser.add_argument("-o", "--output", help='Folder to store output', required=True, type=str)
parser.add_argument('--minreviews', help='Minimum number of reviews', required=True, type=int)
parser.add_argument('--rating', help='Minimum rating', required=True, type=int)

args = vars(parser.parse_args())

def parse_data(data):
    """
    Convert the data to a pandas DataFrame and validate the URLs.

    :param data: A dictionary containing the API response data
    :return: A pandas DataFrame containing the parsed data
    """
    try:
        response_dataframe = pd.DataFrame(data)
        response_list = response_dataframe['gameplay'].tolist()
        response_dataframe = pd.DataFrame(columns=['id', 'url', 'game'])

        for obj in response_list:
            obj_dataframe = pd.DataFrame(obj, index=[0])
            obj_dataframe.rename(columns={"id": "id", "ytUrl": "url", "game": "game"}, inplace=True)
            response_dataframe = pd.concat([response_dataframe, obj_dataframe], ignore_index=True)

        # Validate the URLs
        for row in response_dataframe['url']:
            if not validators.url(row):
                return print("Invalid URL: " + row['url'])

        return response_dataframe
    except Exception as e:
        print(f"Error while parsing data: {e}")
        return pd.DataFrame(columns=['id', 'url', 'game'])

def main():
    """
    Pull URLs from the API that meet the criteria specified in the requirements argument.

    :param requirements: A dictionary containing the requirements for the URLs
    """
    endpoint = args['endpoint']
    params = {
        'rating': args['rating'],
        'minReviews': args['minreviews'],
        "page": 0
    }
    headers =  {'authorization': args['key'], 'authorization_id': args['id']}


    # Make the API request and retrieve the data
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        data = response.json()
        print(data)
        total_pages = data["totalPages"]

        valid_urls = pd.DataFrame(columns=['id', 'url', 'game'])
        for page in range(0, total_pages + 1):  # Query all pages sequentially
            params["page"] = page  # Update page number
            print(f"Requesting page {page}")
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            data = response.json()
            valid_urls = pd.concat([valid_urls, parse_data(data)], ignore_index=True)

        # Filter out duplicate links
        valid_urls.drop_duplicates(subset=["url"], inplace=True)
        download_dir = args['output']

        # Ensure output directory exists
        ensure_dir_exists(Path(download_dir))

        # Save the downloaded links to a file
        valid_urls_df = pd.DataFrame(valid_urls)
        valid_urls_df.to_csv(os.path.join(Path(download_dir), "links.csv"), index=True, columns=["id", "url", "game"])
    except requests.exceptions.Timeout as timeout_error:
        print(f"Request timed out: {timeout_error}")
    except requests.exceptions.TooManyRedirects as redirect_error:
        print(f"Too many redirects: {redirect_error}")
    except requests.exceptions.RequestException as request_error:
        print(f"Request failed: {request_error}")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
