import os
import requests
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
PROXIES = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}

os.makedirs("scraped_data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

visited_links = set()
lock = threading.Lock()
crawled_count = 0  # ✅ Counter for crawled URLs

def check_tor():
    try:
        response = requests.get("https://check.torproject.org/", proxies=PROXIES, timeout=10)
        if "Congratulations" in response.text:
            print("✅ Tor is working correctly!")
            return True
        else:
            print("❌ WARNING: Not using Tor.")
            return False
    except Exception as e:
        print(f"❌ Error checking Tor connection: {e}")
        return False

def extract_onion_links(url):
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        onion_links = set(
            urljoin(url, link['href'])
            for link in soup.find_all('a', href=True)
            if ".onion" in link['href']
        )
        onion_links.update(re.findall(r'(https?://[a-zA-Z0-9]+\.onion\b)', response.text))
        return list(onion_links)
    except:
        return []

def download_content(url, folder_name):
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            extensions = {
                "text/html": "html",
                "application/javascript": "js",
                "text/javascript": "js"
            }
            ext = extensions.get(content_type)
            if not ext:
                return
            filename = os.path.join(folder_name, f"file_{int(time.time() * 1000)}.{ext}")
            with open(filename, 'wb') as f:
                f.write(response.content)
    except:
        pass

def download_assets_parallel(soup, base_url, folder_name):
    tags = soup.find_all(["img", "script", "link", "a"])
    urls = []
    for tag in tags:
        attr = "href" if tag.name in ["link", "a"] else "src"
        if tag.get(attr):
            urls.append(urljoin(base_url, tag[attr]))

    with ThreadPoolExecutor(max_workers = 15) as executor:
        futures = [executor.submit(download_content, u, folder_name) for u in urls]
        for _ in as_completed(futures):
            pass

def crawl_single_site(url):
    global crawled_count

    with lock:
        if url in visited_links:
            return
        visited_links.add(url)

    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=25)
        response.raise_for_status()
    except:
        return

    domain = urlparse(url).netloc.replace('.onion', '')
    folder_name = f"scan/{domain}"
    os.makedirs(folder_name, exist_ok=True)

    with open(os.path.join(folder_name, "index.html"), 'wb') as f:
        f.write(response.content)

    soup = BeautifulSoup(response.content, 'html.parser')
    download_assets_parallel(soup, url, folder_name)

    new_links = extract_onion_links(url)
    with lock:
        for link in new_links:
            if link not in visited_links:
                visited_links.add(link)

        crawled_count += 1
        print(f"\r✅ Crawled: {crawled_count} URLs", end="", flush=True)  # ✅ SINGLE LINE UPDATE


def start_crawl():
    search_engine_url = input("🔍 Enter the site URL to start crawling: ").strip()
    print(f"🚀 Starting crawl at: {search_engine_url}")

    indexed_links = [search_engine_url] + extract_onion_links(search_engine_url)
    if not indexed_links:
        print("❌ No onion links found!")
        return

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(crawl_single_site, link) for link in indexed_links]
        for _ in as_completed(futures):
            pass  # Just wait for all tasks

if __name__ == '__main__':
    if check_tor():
        start_crawl()
