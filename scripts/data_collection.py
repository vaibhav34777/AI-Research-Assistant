import arxiv
import os
import re
import urllib.request

# Ensure the output folder exists
os.makedirs("pdfs", exist_ok=True)

def sanitize(text: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", text)

def download_papers(search, label: str):
    print(f"\n--- Downloading {label} papers ---")
    for result in search.results():
        short_id = result.get_short_id()
        safe_title = sanitize(result.title)[:50]  # first 50 chars
        filename = f"{short_id}_{safe_title}.pdf"
        filepath = os.path.join("pdfs", filename)

        print(f"Downloading [{short_id}] {result.title!r} → {filepath}")
        
        try:
            pdf_url = result.pdf_url
            urllib.request.urlretrieve(pdf_url, filepath)
        except Exception as e:
            print(f"Failed to download {short_id}: {e}")

# Define searches
ml_search = arxiv.Search(
    query="cat:cs.LG OR cat:stat.ML",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

el_search = arxiv.Search(
    query="cat:cs.ET OR cat:eess.ET",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

# Run downloads
download_papers(ml_search, label="Machine Learning")
download_papers(el_search, label="Electronics")

pdf_dir = "pdfs"
for fname in os.listdir(pdf_dir):
    # Only process .pdf files
    if not fname.lower().endswith(".pdf"):
        continue
    
    # Strip only *trailing* whitespace (spaces, tabs) before the extension
    name, ext = os.path.splitext(fname)
    new_name = name.rstrip() + ext  # rstrip() removes trailing whitespace
    if new_name != fname:
        old_path = os.path.join(pdf_dir, fname)
        new_path = os.path.join(pdf_dir, new_name)
        print(f"Renaming:\n  '{fname}'\n→ '{new_name}'")
        os.rename(old_path, new_path)
