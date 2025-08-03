import os
import zipfile
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from google.colab import drive
import shutil
import convert_sfg_to_pgn  # Ensure this is uploaded or pip-installed in Colab

BASE_URL = "https://www.boardspace.net/hive/hivegames/"
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "all_games"

def download_zip_files_from_archive(session, page_url):
    resp = session.get(page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    archive_dirs = [
        urljoin(page_url, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].rstrip("/").startswith("archive-")
    ]

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for arch_url in archive_dirs:
        resp2 = session.get(arch_url)
        resp2.raise_for_status()
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        zip_links = [
            urljoin(arch_url, a["href"])
            for a in soup2.find_all("a", href=True)
            if a["href"].lower().endswith(".zip")
        ]
        for zip_url in zip_links:
            fname = os.path.basename(zip_url)
            outpath = os.path.join(DOWNLOAD_DIR, fname)
            if os.path.exists(outpath):
                continue
            print(f"Downloading {fname}...")
            with session.get(zip_url, stream=True) as dl:
                dl.raise_for_status()
                with open(outpath, "wb") as f:
                    for chunk in dl.iter_content(8192):
                        f.write(chunk)

def unzip_all(download_dir=DOWNLOAD_DIR, extract_dir=EXTRACT_DIR):
    os.makedirs(extract_dir, exist_ok=True)
    for fname in os.listdir(download_dir):
        if fname.lower().endswith(".zip"):
            zip_path = os.path.join(download_dir, fname)
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

def convert_all_sgf(extract_dir=EXTRACT_DIR):
    convert_sfg_to_pgn.main(extract_dir)

def upload_to_gdrive(folder_path):
    print("🔗 Mounting Google Drive...")
    drive.mount('/content/drive')
    dest = f"/content/drive/My Drive/Ortogonale/Hive_DB/"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(folder_path, dest)
    print(f"✅ Uploaded to Google Drive: {dest}")

# ==== MAIN WORKFLOW ====
if __name__ == "__main__":
    with requests.Session() as s:
        download_zip_files_from_archive(s, BASE_URL)
    unzip_all()
    convert_all_sgf()
    upload_to_gdrive(f"pgn-{EXTRACT_DIR}")  # Assuming your converter outputs here
