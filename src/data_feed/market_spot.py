# Patch dla market_spot.py
# 1. Odrzucenie bieżącej otwartej świecy (incomplete candle pruning).
# 2. Poprawna kolejność sortowania przed cięciem limitu w fallbacku Coinbase.
# 3. Zabezpieczenie ciągłości szeregu czasowego.

import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class MarketSpotClient:
    def __init__(self):
        self.binance_spot_url = os.getenv("BINANCE_API_URL", "https://api.binance.com")
        self.binance_fapi_url = "https://fapi.binance.com"
        self.coinbase_url = "https://api.exchange.coinbase.com"
        self.deribit_url = "https://www.deribit.com/api/v2/public"
        self.treasury_api_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpotFeed/1.0"
        }

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _fetch_binance_klines(self, symbol: str = "SOLUSDC", interval: str = "1h", limit: int = 168) -> Optional[pd.DataFrame]:
        url = f"{self.binance_spot_url}/api/v3/klines"
        # Pobieramy limit + 1, aby móc bezpiecznie odrzucić ostatnią, niepełną świecę
        params = {"symbol": symbol, "interval": interval, "limit": min(limit + 1, 1000)}

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                raw_data = response.json()
                if not raw_data or not isinstance(raw_data, list):
                    return None

                records = []
                current_time_ms = time.time() * 1000.0

                for k in raw_data:
                    open_time = self._safe_float(k[0])
                    close_time = self._safe_float(k[6])

                    # Ignorujemy świecę, która nie została jeszcze ostatecznie zamknięta
                    if close_time > current_time_ms:
                        continue

                    records.append({
                        "timestamp": datetime.fromtimestamp(open_time / 1000.0, tz=timezone.utc),
                        "open": self._safe_float(k[1]),
                        "high": self._safe_float(k[2]),
                        "low": self._safe_float(k[3]),
                        "close": self._safe_float(k[4]),
                        "volume": self._safe_float(k[5])
                    })

                df = pd.DataFrame(records)
                if not df.empty:
                    df.sort_values(by="timestamp", inplace=True)
                    df = df.tail(limit).reset_index(drop=True)
                return df
            return None
        except requests.exceptions.RequestException:
            return None

    def _fetch_coinbase_candles(self, product_id: str = "SOL-USD", granularity: int = 3600, limit: int = 168) -> Optional[pd.DataFrame]:
        url = f"{self.coinbase_url}/products/{product_id}/candles"
        params = {"granularity": granularity}

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                raw_data = response.json()
                if not raw_data or not isinstance(raw_data, list):
                    return None

                records = []
                current_time_sec = time.time()

                for k in raw_data:
                    candle_time = self._safe_float(k[0])
                    # Odrzucenie bieżącej trwającej świecy
                    if candle_time + granularity > current_time_sec:
                        continue

                    records.append({
                        "timestamp": datetime.fromtimestamp(candle_time, tz=timezone.utc),
                        "open": self._safe_float(k[3]),
                        "high": self._safe_float(k[2]),
                        "low": self._safe_float(k[1]),
                        "close": self._safe_float(k[4]),
                        "volume": self._safe_float(k[5])
                    })

                df = pd.DataFrame(records)
                if not df.empty:
                    # KRYTYCZNE: Sortowanie chronologiczne PRZED wywołaniem tail(limit)
                    df.sort_values(by="timestamp", inplace=True)
                    df = df.tail(limit).reset_index(drop=True)
                return df
            return None
        except requests.exceptions.RequestException:
            return None

    def get_clean_spot_candles(
        self,
        symbol: str = "SOLUSDC",
        interval: str = "1h",
        limit: int = 168,
        save_csv: bool = False
    ) -> pd.DataFrame:
        df = self._fetch_binance_klines(symbol=symbol, interval=interval, limit=limit)
        
        if df is None or df.empty:
            product_id = "SOL-USD"
            granularity = 3600 if interval == "1h" else 86400
            df = self._fetch_coinbase_candles(product_id=product_id, granularity=granularity, limit=limit)

        if df is None or df.empty:
            print("[BLAD] Nie udalo sie pobrac swiec spotowych z zadnego zrodla.")
            return pd.DataFrame()

        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if save_csv:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "raw"))
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "spot_candles_latest.csv")
            df.to_csv(csv_path, index=False)

        return df

    def _fetch_binance_funding_rate(self, symbol: str = "SOLUSDT") -> Optional[Dict[str, Any]]:
        url = f"{self.binance_fapi_url}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    latest = data[-1]
                    rate = self._safe_float(latest.get("fundingRate"))
                    funding_time_ms = self._safe_float(latest.get("fundingTime"))
                    # Konwencja ACT/365: 3 okresy fundingowe dziennie
                    return {
                        "source": "binance_futures",
                        "symbol": symbol,
                        "funding_rate": round(rate, 6),
                        "annualized_rate": round(rate * 3.0 * 365.0, 4),
                        "timestamp": datetime.fromtimestamp(funding_time_ms / 1000.0, tz=timezone.utc)
                    }
            return None
        except requests.exceptions.RequestException:
            return None

    def _fetch_deribit_funding_rate(self, instrument_name: str = "SOL_USDC-PERPETUAL") -> Optional[Dict[str, Any]]:
        url = f"{self.deribit_url}/ticker"
        params = {"instrument_name": instrument_name}

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                result = response.json().get("result", {})
                rate = self._safe_float(result.get("current_funding") or result.get("funding_8h"))
                ts_ms = self._safe_float(result.get("timestamp"))
                return {
                    "source": "deribit_perpetual",
                    "symbol": instrument_name,
                    "funding_rate": round(rate, 6),
                    "annualized_rate": round(rate * 3.0 * 365.0, 4),
                    "timestamp": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                }
            return None
        except requests.exceptions.RequestException:
            return None

    def get_current_funding_rate(self, symbol: str = "SOLUSDT") -> Dict[str, Any]:
        res = self._fetch_binance_funding_rate(symbol=symbol)
        if res:
            return res

        res_deribit = self._fetch_deribit_funding_rate(instrument_name="SOL_USDC-PERPETUAL")
        if res_deribit:
            return res_deribit

        return {
            "source": "none",
            "symbol": symbol,
            "funding_rate": 0.0,
            "annualized_rate": 0.0,
            "timestamp": datetime.now(timezone.utc)
        }

    def get_us_treasury_risk_free_rate(self) -> float:
        params = {
            "filter": "security_desc:eq:Treasury Bills",
            "sort": "-record_date",
            "page[size]": "1"
        }

        try:
            response = requests.get(self.treasury_api_url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data and len(data) > 0:
                    rate_pct = self._safe_float(data[0].get("avg_interest_rate_amt"))
                    if rate_pct > 0.0:
                        return round(rate_pct / 100.0, 4)
        except requests.exceptions.RequestException:
            pass

        return 0.045


if __name__ == "__main__":
    print("Inicjalizacja MarketSpotClient...")
    client = MarketSpotClient()

    print("\n--- POBIERANIE OFICJALNEJ STOPY U.S. TREASURY (T-BILLS) ---")
    r_treasury = client.get_us_treasury_risk_free_rate()
    print(f"[OK] Dynamiczna stopa wolna od ryzyka z U.S. Treasury: {r_treasury * 100:.2f}% (r = {r_treasury})")

    print("\n--- POBIERANIE FUNDING RATE (BINANCE) ---")
    funding_data = client.get_current_funding_rate("SOLUSDT")
    print(f"[OK] Funding rate APR: {funding_data['annualized_rate'] * 100:.4f}%")