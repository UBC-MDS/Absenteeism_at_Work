# author: Yiki Su
# date: 2020-11-20

"""The sript downloads the zip file from the web to a local filepath and unzip the file. Save it in a local path.
Usage: downoad_data_failed.py --url=<url> --out_path=<out_path>

Options:
--url=<url>              URL from where to download the zip file
--out_path=<out_path>    Path of where to locally save the file
"""
  
from docopt import docopt
import requests
import os
import pandas as pd
import io
import zipfile
from zipfile import ZipFile

opt = docopt(__doc__)

def main(url, out_path):
  try: 
    request = requests.get(url)
    request.status_code == 200
  except Exception as req:
    print("Website at the provided url does not exist.")
    print(req)
    
  zipDocument = zipfile.ZipFile(io.BytesIO(request.content))
  zipDocument.extractall(path=out_path)

if __name__ == "__main__":
  main(opt["--url"], opt["--out_path"])
