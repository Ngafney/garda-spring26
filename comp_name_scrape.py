import os
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def fetch_sector(company_name):
    """Worker function to find the ticker and sector for a single company."""
    clean_name = company_name.replace('-', ' ').replace('_', ' ')
    sector = "Unassigned"
    ticker = None
    
    # 1. Search Yahoo for the ticker
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_name}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            ticker = data['quotes'][0]['symbol']
            
            # 2. If ticker is found, grab the sector
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'Unassigned')
    except Exception:
        pass # Silently fail on network drops or unfound tickers
        
    return {"company_slug": company_name, "ticker": ticker, "sector": sector}

def main():
    folder_dir = r"C:\Users\rajaa\OneDrive\Desktop\GardDat"
    list_path = os.path.join(folder_dir, "company_list.txt")
    output_path = os.path.join(folder_dir, "company_sector_map.csv")
    
    # Read the companies
    if not os.path.exists(list_path):
        print(f"❌ Could not find {list_path}")
        return
        
    with open(list_path, 'r') as f:
        companies = [line.strip() for line in f if line.strip()]
        
    print(f"⚡ Turbo-Fetching sectors for {len(companies)} companies...")
    
    results = []
    
    # MULTITHREADING: Process 15 companies at the same time
    # (Don't set this higher than 20, or Yahoo might temporarily block your IP)
    MAX_THREADS = 15 
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Submit all tasks to the thread pool
        future_to_company = {executor.submit(fetch_sector, comp): comp for comp in companies}
        
        # Use tqdm to show a progress bar as tasks complete
        for future in tqdm(as_completed(future_to_company), total=len(companies), desc="Mapping"):
            results.append(future.result())
            
    # Save the master mapping dictionary as a CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✅ DONE in record time! Sector map saved to: {output_path}")

if __name__ == "__main__":
    main()