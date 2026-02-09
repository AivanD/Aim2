import time
import asyncio
from tqdm.asyncio import tqdm
import logging
import aiohttp
import os

INPUT_CSV = "/home/aivan-dolor/Documents/Aim2/tardc_ids_dedup.csv"
INPUT_LIST_OF_PMCIDS = "/home/aivan-dolor/Documents/Aim2/tardc_pmc_ids.txt"
OUTPUT_DIR = "/home/aivan-dolor/Documents/Aim2/tardc/input/"
API_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/"
UNSUCCESSFUL_LOG = os.path.join(OUTPUT_DIR, "unsuccessful_downloads.txt")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_pmc_article(session: aiohttp.ClientSession, pmcid: str, semaphore: asyncio.Semaphore) -> tuple[str, bool]:
    """
    Fetches an article from the PMC API, handles rate limiting, and saves it to a file.
    Returns a tuple of (pmcid, success_status).
    """
    url = f"{API_BASE_URL}{pmcid}/ascii"
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    output_path = os.path.join(OUTPUT_DIR, f"{pmcid}.xml")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    # logger.info(f"Successfully downloaded and saved {pmcid}.xml")
                    return (pmcid, True)
                elif response.status == 429: # Too Many Requests
                    retry_after = int(response.headers.get("Retry-After", "3"))
                    logger.warning(f"Rate limit reached. Waiting for {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    # Retry the request
                    return await fetch_pmc_article(session, pmcid, semaphore)
                else:
                    logger.error(f"Failed to download {pmcid}. Status: {response.status}")
                    return (pmcid, False)
        except aiohttp.ClientError as e:
            logger.error(f"Network error for {pmcid}: {e}")
            return (pmcid, False)

def process_csv():
    """
    Processes the input CSV file to extract unique PMCIDs and saves them to a text file.
    """
    import pandas as pd

    try:
        df = pd.read_csv(INPUT_CSV)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
        if 'id' not in df.columns:
            logger.error(f"'id' column not found in {INPUT_CSV}")
            return
        
        # Strip whitespace from 'id' column values and then get unique values
        unique_pmcids = df['id'].str.strip().dropna().unique()
        
        with open(INPUT_LIST_OF_PMCIDS, "w") as f:
            for pmcid in unique_pmcids:
                f.write(f"{pmcid}\n")
        logger.info(f"Extracted {len(unique_pmcids)} unique PMCIDs to {INPUT_LIST_OF_PMCIDS}")
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")

async def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # process_csv()
    
    semaphore = asyncio.Semaphore(5) # Limit concurrent requests to 3
    
    try:
        with open(INPUT_LIST_OF_PMCIDS, "r") as f:
            pmcids = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logger.error(f"Input file not found: {INPUT_LIST_OF_PMCIDS}")
        return

    unsuccessful_pmcids = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_pmc_article(session, pmcid, semaphore) for pmcid in pmcids]
        results = await tqdm.gather(*tasks, desc="Downloading PMC articles")

    for pmcid, success in results:
        if not success:
            unsuccessful_pmcids.append(pmcid)

    if unsuccessful_pmcids:
        logger.warning(f"Found {len(unsuccessful_pmcids)} unsuccessful downloads.")
        with open(UNSUCCESSFUL_LOG, "w") as f:
            for pmcid in unsuccessful_pmcids:
                f.write(f"{pmcid}\n")
        logger.info(f"List of unsuccessful PMCIDs saved to {UNSUCCESSFUL_LOG}")
    else:
        logger.info("All articles downloaded successfully.")

if __name__ == "__main__":
    asyncio.run(main())