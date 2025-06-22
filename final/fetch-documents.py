import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import os

def crawl_website(start_url, output_filename="crawled_documents.txt", max_pages=100):
    """
    Crawls a website starting from a given URL, extracts text content and links,
    and saves them to a text file.

    Args:
        start_url (str): The starting URL for the crawl.
        output_filename (str): The name of the file to save the extracted content.
        max_pages (int): The maximum number of pages to crawl to prevent infinite loops.
    """
    base_domain = urlparse(start_url).netloc
    if not base_domain:
        print(f"Error: Could not parse domain from start URL: {start_url}")
        return

    # Use a deque for efficient append/pop operations
    urls_to_visit = deque([start_url])
    visited_urls = set()
    crawled_count = 0

    print(f"Starting crawl from: {start_url}")
    print(f"Saving content to: {output_filename}")

    # Create or clear the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("") # Clear file content if it exists

    # User-Agent to mimic a browser, helps avoid some blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while urls_to_visit and crawled_count < max_pages:
        current_url = urls_to_visit.popleft()

        # Normalize URL by removing fragments and query parameters for consistent tracking
        parsed_current_url = urlparse(current_url)
        normalized_current_url = urljoin(current_url, parsed_current_url.path)

        if normalized_current_url in visited_urls:
            continue

        if base_domain not in normalized_current_url:
            # Skip external links
            continue

        print(f"Crawling ({crawled_count + 1}/{max_pages}): {normalized_current_url}")
        visited_urls.add(normalized_current_url)
        crawled_count += 1

        try:
            response = requests.get(normalized_current_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            soup = BeautifulSoup(response.text, 'html.parser')

            # --- Extract Main Document Text ---
            # This part might need adjustment depending on the exact HTML structure
            # of the target website's content area. Common areas are:
            # - div with specific classes (e.g., 'markdown', 'content', 'main-content')
            # - article tag
            # - main tag
            # For docs.apiculus.com, 'theme-doc-markdown markdown' seems relevant.
            content_div = soup.find('div', class_='theme-doc-markdown markdown')
            if not content_div:
                # Fallback to other common content areas if the specific div is not found
                content_div = soup.find('article') or soup.find('main')

            document_text = []
            if content_div:
                # Extract text from paragraph tags within the content area
                for p_tag in content_div.find_all('p'):
                    cleaned_text = p_tag.get_text(strip=True)
                    if cleaned_text: # Only add non-empty paragraphs
                        document_text.append(cleaned_text)
                # Also try to get text from headings (h1-h6) and list items (li)
                for tag in content_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    cleaned_text = tag.get_text(strip=True)
                    if cleaned_text and cleaned_text not in document_text: # Avoid duplicates if p_tag already got it
                        document_text.append(cleaned_text)

            # If no specific content_div found, try to get all visible text from body
            if not document_text:
                body_text = soup.body.get_text(separator='\n', strip=True)
                # Simple heuristic to get "meaningful" lines
                document_text = [line for line in body_text.split('\n') if line.strip() and len(line.strip()) > 20]


            # --- Write to Output File ---
            with open(output_filename, 'a', encoding='utf-8') as f:
                f.write(f"{normalized_current_url}\n")
                # f.write(f"Document URL - {normalized_current_url}\n")
                # if document_text:
                #     f.write("\n".join(document_text))
                # else:
                    # f.write("<No main content found for this URL>")
                #f.write("\n\n---\n\n") # Separator for clarity


            # --- Discover New Links ---
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(normalized_current_url, href)
                parsed_absolute_url = urlparse(absolute_url)
                normalized_absolute_url = urljoin(absolute_url, parsed_absolute_url.path) # Remove query/fragment

                # Ensure it's an internal link and not yet visited
                if base_domain in normalized_absolute_url and \
                   normalized_absolute_url not in visited_urls and \
                   normalized_absolute_url not in urls_to_visit:
                    urls_to_visit.append(normalized_absolute_url)

        except requests.exceptions.RequestException as e:
            print(f"Error crawling {normalized_current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {normalized_current_url}: {e}")

    print(f"\nCrawl finished. Extracted content from {crawled_count} pages.")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    start_url = "https://docs.apiculus.com/docs/intro" # Changed to a plain URL string
    crawl_website(start_url, max_pages=50000) # You can adjust max_pages as needed
