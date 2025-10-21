import time
import re
import numpy as np
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import streamlit as st

def get_most_replayed_timestamps(youtube_url, video_duration, logger):
    """Scrapes a YouTube page to find the most replayed sections."""
    logger.info(f"Starting YouTube scrape for 'most replayed' on {youtube_url}")
    st.write("Scraping YouTube for 'most replayed' sections...")
    
    chrome_options = Options()
    if os.environ.get("HEADLESS", "false").lower() == "true":
        logger.info("Scraper running in headless mode.")
        st.info("Running scraper in headless mode.")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Use the pre-installed chromedriver in the Docker container
    service = Service(executable_path="/usr/bin/chromedriver")
    driver = None
    try:
        logger.info("Initializing ChromeDriver with pre-installed binary...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(youtube_url)
        logger.info("Page loaded. Waiting for dynamic elements...")
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        path_element = soup.find("path", {"class": "ytp-heat-map-path"})

        if not path_element:
            logger.warning("Could not find 'most replayed' SVG path element on the page.")
            st.warning("Could not find 'most replayed' data on the page. This feature may not be available for all videos.")
            return []

        d_attribute = path_element.get('d')
        logger.info("Parsing SVG path data.")
        
        points = re.findall(r'C\s*[\d\.]+,([\d\.]+)\s*[\d\.]+,([\d\.]+)\s*[\d\.]+,(.+?)\s', d_attribute)
        y_coords = [float(p[2]) for p in points]
        
        if not y_coords:
            logger.warning("Failed to parse heat map coordinates from SVG path."); return []

        timestamps = np.linspace(0, video_duration, len(y_coords))
        avg_y = np.mean(y_coords); std_y = np.std(y_coords)
        threshold = avg_y - 1.0 * std_y
        peak_indices = np.where(np.array(y_coords) < threshold)[0]
        
        if len(peak_indices) == 0:
            logger.info("No significant replay peaks found."); return []

        significant_timestamps = []
        last_index = -10
        for idx in peak_indices:
            if idx > last_index + 5: significant_timestamps.append(timestamps[idx])
            last_index = idx

        logger.info(f"Found {len(significant_timestamps)} high-engagement timestamps.")
        st.write(f"Found {len(significant_timestamps)} high-engagement timestamps.")
        return significant_timestamps

    except Exception as e:
        logger.exception("An error occurred during YouTube scraping.")
        st.error(f"An error occurred during scraping: {e}")
        return []
    finally:
        if driver:
            driver.quit()
