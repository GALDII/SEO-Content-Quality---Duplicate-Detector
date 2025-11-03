"""
HTML Parsing and Web Scraping utilities
"""
from bs4 import BeautifulSoup
import requests
from time import sleep

def scrape_url(url, timeout=10):
    """
    Scrape HTML content from a URL.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        str: HTML content or empty string on failure
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

def parse_html_content(html_content):
    """
    Parse HTML content and extract meaningful text.
    
    Args:
        html_content: Raw HTML string
        
    Returns:
        dict: Extracted information (title, body_text, word_count)
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # Extract main content from various tags
        content_tags = ['article', 'main', 'div', 'section']
        body_text = ""
        
        for tag in content_tags:
            elements = soup.find_all(tag)
            if elements:
                for elem in elements:
                    # Get all paragraph texts
                    paragraphs = elem.find_all('p')
                    if paragraphs:
                        body_text += " ".join([p.get_text(strip=True) for p in paragraphs])
                if body_text:
                    break
        
        # Fallback: extract all paragraph tags
        if not body_text:
            paragraphs = soup.find_all('p')
            body_text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Clean text
        body_text = " ".join(body_text.split())
        
        # Calculate word count
        word_count = len(body_text.split())
        
        return {
            'title': title,
            'body_text': body_text,
            'word_count': word_count
        }
        
    except Exception as e:
        print(f"Error parsing HTML: {str(e)}")
        return {
            'title': "",
            'body_text': "",
            'word_count': 0
        }