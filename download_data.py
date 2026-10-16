"""
Helper script to download volunteer_data_en.csv from GitHub repo
"""
import argparse
import os
import urllib.request

URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
OUT = "volunteer_data_en.csv"

def download(force=False):
    if os.path.exists(OUT) and not force:
        size_mb = os.path.getsize(OUT) / (1024 * 1024)
        print(f"'{OUT}' already exists ({size_mb:.2f} MB). Use --force to re-download.")
        return
    print(f"Downloading {OUT} from {URL} ...")
    urllib.request.urlretrieve(URL, OUT)
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"Downloaded successfully: {OUT} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download latest volunteer_data_en.csv")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if file exists")
    args = parser.parse_args()
    download(force=args.force)
