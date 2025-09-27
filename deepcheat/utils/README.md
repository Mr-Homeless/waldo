# utils
This directory contains all the preprocessing and utility scripts used across the models repo.

### install-environment.sh
---
This shell script can be run to automatically create an anaconda environment with the dependencies installed.
The full list of dependencies can be found in models/environment.yml

### generate-requirements.sh
---
After activating a conda environment this shell script can be run to export environment.yml

### link-retrieval.py
---
Get URL's from API and stores them locally
- Options
  - -e or --endpoint is the target url on the server
    - Default: `https://waldo.vision/api/trpc/urls`
    - Required: `False`
  - -k or --key is the API Key
    - If left unspecified the default key will be pulled from the environment variable `WALDO_API_KEY`
    - Required: `False`
  - -o or --output is the folder where the file will be stored
    - Required: `True`
    - Example: `/home/usr/website/files/` the output would then be `/home/usr/website/files/links.csv`
  - -r or --requirements is the specifications for the links you want to download
    - Default: `{"minReviews": 25, "rating": 90}`
    - Example: `{"minReviews": 20, "rating": 95, "otherspecfication": "gamename"}`

### download_links.py
---
Download YouTube videos using the CSV output of link-retrieval.py
- Options
  - -i or --input is the path to the CSV of YouTube URLs to be downloaded
    - Required: `True`
  - -o or --output is the path to the folder where downloaded videos will be placed
    - Required: `True`
  - -r or --max_resolution is the maximum resolution of video that the script will try to download
    - Default: `1080`
    - Required: `False`