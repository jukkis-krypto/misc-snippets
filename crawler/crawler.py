#!/usr/bin/env python3
"""
Crawler for extracting text content from websites within the same domain
"""

import requests
import re
from urllib.request import Request, urlopen
from urllib.parse import urlparse, quote, unquote
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from collections import deque
import os
import logging
import yaml
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as configfile:
    config = yaml.safe_load(configfile)
    domain = config['domain']
    initial_path = config['initial_path']
    page_limit = config.get('page_limit', None)
    stay_within_path = config.get('stay_within_path', False)
    count_only = config.get('count_only', False)
    restricted_words = config['restricted_words']

# Combine all restricted words into a single list
excluded_substrings = []
for category in restricted_words.values():
    excluded_substrings.extend(category)

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

class HyperlinkParser(HTMLParser):
    """ HTML Parser to extract hyperlinks """

    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        """ Override HTMLParser's handle_starttag method """
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

def get_hyperlinks(url):
    """ Extract hyperlinks from a URL """
    try:
        req = Request(quote(url, safe=":/?#[]@!$&'()*+,;="))
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0')
        with urlopen(req) as response:
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            html = response.read().decode('utf-8')  # Decode using UTF-8
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}\n{traceback.format_exc()}")
        return []

    parser = HyperlinkParser()
    parser.feed(html)
    return parser.hyperlinks

def get_domain_hyperlinks(local_domain, url, base_path=None):
    """ Get hyperlinks within the same domain and optionally within the same path """
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                if base_path and not url_obj.path.startswith(base_path):
                    continue
                clean_link = link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = f"https://{local_domain}/{link}"

            if base_path and not clean_link.startswith(f"https://{local_domain}{base_path}"):
                continue

        if clean_link:
            clean_link = unquote(clean_link.rstrip('/'))
            clean_links.append(clean_link)
    return list(set(clean_links))

def sanitize_filename(url):
    """Sanitize the filename to handle special characters"""
    return quote(url[8:].replace("/", "_"), safe="")

def clean_html(raw_html):
    """Clean HTML content to text, extracting only content inside element with id='content-main'"""
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # Find the element with id="content-main"
    # Hardcoded content filter - TODO: move it to config
    content_main = soup.find(id="main")
    
    # If the element is found, process its text content
    if content_main:
        text = ""
        for element in content_main.recursiveChildGenerator():
            if element.name in ['p', 'br', 'div']:
                text += "\n"
            elif isinstance(element, str):
                text += element.strip()
        return text.strip()
    
    # Return an empty string if the element is not found
    return ""

def crawl(url, limit=None, stay_within_path=False, count_only=False):
    """ Crawl a URL with an optional limit on the number of pages """
    local_domain = urlparse(url).netloc
    base_path = urlparse(url).path if stay_within_path else None
    queue = deque([url])
    seen = set([url])
    pages_crawled = 0

    if not count_only:
        os.makedirs(f"data/raw/{local_domain}/", exist_ok=True)

    while queue:
        if limit and pages_crawled >= limit:
            logger.info(f"Reached the page limit of {limit}. Stopping crawl.")
            break

        url = queue.pop()
        logger.info(f"({pages_crawled + 1}) Crawling: {url}")

        if not count_only:
            try:
                sanitized_filename = sanitize_filename(url)
                # Write the content to a file, the file will later be splitted into chunks,
                # chunks written into db and vector embeddings generated for each chunk
                with open(f'data/raw/{local_domain}/{sanitized_filename}.txt', "w", encoding="UTF-8") as f:
                    headers = requests.utils.default_headers()
                    headers.update({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                    })
                    response = requests.get(url, headers=headers)
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract the title of the page
                    title = soup.title.string if soup.title else "No title"
                    
                    # Write the source URL and title to top of the file for later reference
                    f.write(f"Source URL: {url}\n")
                    f.write(f"Title: {title}\n\n")
                    
                    # Clean and write the main content
                    text = clean_html(soup.prettify())
                    if "You need to enable JavaScript to run this app." in text:
                        logger.warning(f"JavaScript required for {url}")
                    else:
                        f.write(text)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

        pages_crawled += 1

        for link in get_domain_hyperlinks(local_domain, url, base_path):
            if link not in seen:
                if not any(sub in link for sub in excluded_substrings):
                    queue.append(link)
                seen.add(link)

    logger.info(f"Total pages counted: {pages_crawled}")

# Ensure this part is executed only when the script is run directly
if __name__ == "__main__":
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    with open('config.yaml', 'r') as configfile:
        config = yaml.safe_load(configfile)
        initial_path = config['initial_path']
        page_limit = config.get('page_limit', None)
        stay_within_path = config.get('stay_within_path', False)
        count_only = config.get('count_only', False)

    logger.info(f"Starting crawl for {initial_path}")
    crawl(initial_path, limit=page_limit, stay_within_path=stay_within_path, count_only=count_only)
    logger.info("Crawl finished")
