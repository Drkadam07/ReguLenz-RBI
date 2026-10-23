# import os
# from pathlib import Path
# import httpx
# from tqdm import tqdm

# RAW_DATA_DIR = Path("data/raw")
# RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# # Curated starter circulars directly accessible from RBI's document delivery server
# SEED_DOCUMENTS = {
#     "Master_Direction_KYC.pdf": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/18MDKYCD8A6F8A2D3C04A929286C3E481232871.PDF",
#     "Digital_Lending_Guidelines_2022.pdf": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/GUIDELINESDIGITALLENDINGD54E9130D03A4E21A467616BE3D446CA.PDF",
#     "Fair_Lending_Practice_Penal_Charges.pdf": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/77MD180820239A06041797ED4E958055620959A1CA76.PDF",
#     "Cyber_Security_Framework_Banks.pdf": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/CSFB0206168F23A3AC2137452FA7BF4C5A510762BF.PDF"
# }

# def download_corpus():
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }
    
#     with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
#         for filename, url in SEED_DOCUMENTS.items():
#             dest = RAW_DATA_DIR / filename
#             if dest.exists():
#                 print(f"Skipping {filename} (already exists)")
#                 continue
                
#             print(f"Downloading {filename}...")
#             try:
#                 response = client.get(url)
#                 response.raise_for_status()
#                 with open(dest, "wb") as f:
#                     f.write(response.content)
#                 print(f"Saved: {filename} ({len(response.content) // 1024} KB)")
#             except Exception as e:
#                 print(f"Failed to fetch {url}: {e}")

# if __name__ == "__main__":
#     download_corpus()