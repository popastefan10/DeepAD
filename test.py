import zipfile
with zipfile.ZipFile("./data/processed/DAGM/4ppi_176px_pad_v2.zip", 'r') as zip_ref:
    zip_ref.extractall("./data/processed/DAGM/")