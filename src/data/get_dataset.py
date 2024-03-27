import requests
import zipfile
import os

def fetch_and_extract_from_zenodo(doi, extract_dir):
    """
    Fetches a zip file from Zenodo for a given DOI and extracts its contents into a specified directory.
    
    Args:
    - doi (str): Digital Object Identifier of the Zenodo record.
    - extract_dir (str): Directory where the contents of the zip file will be extracted.
    
    Returns:
    - bool: True if extraction is successful, False otherwise.
    """
    base_url = "https://zenodo.org/api/records/"
    url = base_url + doi
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        files = data['files']
        for file_info in files:
            if file_info['type'] == 'zip':
                download_url = file_info['links']['self']
                zip_file_path = os.path.join(extract_dir, doi.replace("/", "_") + ".zip")
                # Download the zip file
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(zip_file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                # Extract the contents of the zip file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                # Remove the downloaded zip file
                os.remove(zip_file_path)
                return True
        print("No zip file found in the Zenodo record.")
        return False
    else:
        print(f"Error fetching data from Zenodo. Status code: {response.status_code}")
        return False


if __name__ == '__main__':
    doi = "10.5281/zenodo.10855059"  
    data_directory = os.path.join('..','data','raw')
    fetch_and_extract_from_zenodo(doi, data_directory)
