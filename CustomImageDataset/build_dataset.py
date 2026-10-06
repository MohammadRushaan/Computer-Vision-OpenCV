# build_dataset.py

# ==============================================================================
# 1. IMPORT REQUIRED PACKAGES
# ==============================================================================
import os                # Directory and path manipulation
import argparse          # CLI argument parser
import cv2               # OpenCV for verifying image integrity
from icrawler.builtin import BingImageCrawler  # Resilient multi-threaded crawler

# ==============================================================================
# 2. PARSE COMMAND-LINE ARGUMENTS
# ==============================================================================
ap = argparse.ArgumentParser(description="Reliable Multi-Threaded Dataset Scraper")
ap.add_argument("-q", "--query", required=True, 
                help="Search keyword (e.g. 'pikachu', 'charmander')")
ap.add_argument("-o", "--output", required=True, 
                help="Target directory path to store downloaded images")
ap.add_argument("-m", "--max-results", type=int, default=100, 
                help="Maximum number of images to download")
args = vars(ap.parse_args())

# Ensure output directory exists
os.makedirs(args["output"], exist_ok=True)

# ==============================================================================
# 3. INITIALIZE AND EXECUTE CRAWLER
# ==============================================================================
print(f"[INFO] Crawling images for query: '{args['query']}'...")

# Bing crawler automatically rotates user-agents and avoids rate-limit blocks
crawler = BingImageCrawler(
    downloader_threads=4,               # 4 concurrent download threads
    storage={"root_dir": args["output"]}
)

# Start crawl
crawler.crawl(
    keyword=args["query"], 
    max_num=args["max_results"]
)

# ==============================================================================
# 4. OPENCV INTEGRITY PRUNING
# ==============================================================================
print("\n[INFO] Validating image integrity with OpenCV...")
valid_count = 0
pruned_count = 0

for root, _, filenames in os.walk(args["output"]):
    for filename in filenames:
        filepath = os.path.join(root, filename)

        # Attempt to decode image with OpenCV
        image = cv2.imread(filepath)

        # If OpenCV fails to read the image (corrupt file, 404 HTML blob, etc.)
        if image is None:
            print(f"[REMOVED] Corrupt/unreadable file: {filepath}")
            os.remove(filepath)
            pruned_count += 1
        else:
            valid_count += 1

print(f"\n[INFO] Dataset Summary for '{args['query']}':")
print(f"  - Valid Images: {valid_count}")
print(f"  - Pruned Corrupt Images: {pruned_count}")
print(f"  - Saved Location: {os.path.abspath(args['output'])}")