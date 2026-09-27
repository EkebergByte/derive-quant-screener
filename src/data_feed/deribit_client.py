# Patch dla deribit_client.py
# 1. Zastąpienie _safe_float mechanizmem _safe_optional_float dla bid/ask/iv.
# 2. Ujednolicenie ACT/365.0.
# 3. Zabezpieczenie ścieżki zapisu pliku.

import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DeribitClient:
    MONTH_MAP = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
    }

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("DERIBIT_API_URL", "https://www.deribit.com/api/v2")
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeribitQuant/1.0"
        }

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_optional_float(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _send_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/public/{endpoint}"
        try:
            response = requests.get(url, params=params or {}, headers=self.headers, timeout=12)
            response.raise_for_status()
            data = response.json()
            return data.get("result")
        except requests.exceptions.RequestException as e:
            print(f"[BLAD DERIBIT] {url} -> {e}")
            return None

    def get_spot_price(self, underlying: str = "SOL") -> Optional[float]:
        index_name = f"{underlying.lower()}_usdc"
        res = self._send_get("get_index_price", {"index_name": index_name})
        if res and isinstance(res, dict) and "index_price" in res:
            return self._safe_float(res["index_price"])
        return None

    def get_historical_volatility(self, currency: str = "SOL") -> Optional[float]:
        res = self._send_get("get_historical_volatility", {"currency": currency})
        if res and isinstance(res, list) and len(res) > 0:
            latest_point = res[-1]
            if len(latest_point) >= 2:
                raw_vol = self._safe_float(latest_point[1])
                return round(raw_vol / 100.0, 4)
        return None

    def parse_deribit_instrument_name(self, name: str) -> Optional[Dict[str, Any]]:
        pattern = r"^([A-Z]+_[A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])$"
        match = re.match(pattern, name.upper())
        if not match:
            return None

        pair, day_str, mon_str, yr_str, strike_str, opt_char = match.groups()
        mon_num = self.MONTH_MAP.get(mon_str)
        if not mon_num:
            return None

        full_year = f"20{yr_str}"
        full_day = f"{int(day_str):02d}"
        expiry_str = f"{full_year}{mon_num}{full_day}"

        return {
            "pair": pair,
            "expiry_str": expiry_str,
            "strike": self._safe_float(strike_str),
            "option_type": "CALL" if opt_char == "C" else "PUT"
        }

    def get_clean_options_chain(
        self,
        underlying: str = "SOL",
        min_dte: float = 7.0,
        max_dte: float = 35.0,
        save_csv: bool = False
    ) -> pd.DataFrame:
        raw_items = self._send_get("get_book_summary_by_currency", {
            "currency": "USDC",
            "kind": "option"
        })

        if not raw_items or not isinstance(raw_items, list):
            return pd.DataFrame()

        current_time_utc = datetime.now(timezone.utc)
        target_prefix = f"{underlying.upper()}_USDC-"
        fallback_spot = self.get_spot_price(underlying=underlying)

        records = []

        for item in raw_items:
            name = item.get("instrument_name", "")
            if not name.startswith(target_prefix):
                continue

            parsed = self.parse_deribit_instrument_name(name)
            if not parsed:
                continue

            expiry_str = parsed["expiry_str"]

            try:
                exp_dt = datetime.strptime(f"{expiry_str} 08:00:00", "%Y%m%d %H:%M:%S").replace(tzinfo=timezone.utc)
                seconds_to_expiry = (exp_dt - current_time_utc).total_seconds()
            except ValueError:
                continue

            if seconds_to_expiry <= 0:
                continue

            dte = seconds_to_expiry / 86400.0
            if not (min_dte <= dte <= max_dte):
                continue

            # Standardowa rynkowa konwencja ACT/365.0
            t_annual = seconds_to_expiry / (365.0 * 86400.0)

            spot_price = self._safe_float(item.get("underlying_price") or item.get("estimated_delivery_price") or fallback_spot)
            mark_price = self._safe_float(item.get("mark_price"))
            
            # Prawidłowa obsługa braku kwotowań (None zamiast 0.0)
            bid_price = self._safe_optional_float(item.get("bid_price"))
            ask_price = self._safe_optional_float(item.get("ask_price"))
            
            raw_mark_iv = self._safe_optional_float(item.get("mark_iv"))
            mark_iv = round(raw_mark_iv / 100.0, 4) if raw_mark_iv is not None else None

            oi = self._safe_float(item.get("open_interest"))
            volume = self._safe_float(item.get("volume"))

            records.append({
                "instrument_name": name,
                "currency": underlying,
                "expiry_str": expiry_str,
                "dte": round(dte, 4),
                "T": round(t_annual, 6),
                "strike": parsed["strike"],
                "option_type": parsed["option_type"],
                "spot_price": spot_price,
                "bid": bid_price,
                "ask": ask_price,
                "mark": mark_price,
                "iv": mark_iv,
                "oi": oi,
                "volume_24h": volume
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by=["dte", "strike", "option_type"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            if save_csv:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "raw"))
                os.makedirs(output_dir, exist_ok=True)
                csv_path = os.path.join(output_dir, "deribit_sol_options_latest.csv")
                df.to_csv(csv_path, index=False)

        return df


if __name__ == "__main__":
    print("Inicjalizacja oczyszczonego klienta Deribit...")
    client = DeribitClient()

    spot = client.get_spot_price("SOL")
    print(f"[OK] Kurs indeksowy SOL z Deribit: {spot} USD")

    rv = client.get_historical_volatility("SOL")
    if rv:
        print(f"[OK] Oficjalna kroczaca zmiennosc zrealizowana (RV) SOL: {rv * 100:.2f}%")

    MIN_DTE = 7.0
    MAX_DTE = 35.0

    print(f"\nPobieranie benchmarkowych opcji na SOL z Deribit (DTE: {MIN_DTE} - {MAX_DTE} dni)...")
    df_deribit = client.get_clean_options_chain("SOL", min_dte=MIN_DTE, max_dte=MAX_DTE, save_csv=True)

    print(f"[SUKCES] Pobrano {len(df_deribit)} czystych opcji z Deribit!")

    if not df_deribit.empty:
        expiries = df_deribit["expiry_str"].unique()
        print(f"Dostepne serie wygasania: {list(expiries)}")

        first_exp = expiries[0]
        print(f"\nOpcje At-The-Money (Strike 115 - 125 USD) dla serii {first_exp}:")
        atm_view = df_deribit[
            (df_deribit["expiry_str"] == first_exp) &
            (df_deribit["strike"] >= 115) & 
            (df_deribit["strike"] <= 125)
        ]
        print(atm_view[["instrument_name", "strike", "option_type", "bid", "ask", "mark", "iv", "oi", "volume_24h"]])