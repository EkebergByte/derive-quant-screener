import os
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DeriveClient:
    """
    Autonomiczny, defensywny klient danych rynkowych dla gieldy Derive.xyz (Lyra v2/v3).
    Dziala bezobslugowo w chmurze, mapuje dynamiczny kalendarz DTE i przetwarza jednoliterowa telemetrie.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("DERIVE_API_BASE_URL", "https://api.lyra.finance")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeriveQuant/1.0"
        }

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _send_post(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/public/{method}"
        payload = params or {}

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=12)
            response.raise_for_status()
            data = response.json()
            
            if "result" in data:
                return data["result"]
            return data
        except requests.exceptions.RequestException as e:
            print(f"[BLAD SIECIOWY] {url} -> {e}")
            return None

    def get_spot_price(self, currency: str = "SOL") -> Optional[float]:
        """
        Pobiera biezacy kurs referencyjny z kontraktu wieczystego SOL-PERP.
        """
        ticker = self._send_post("get_ticker", {"instrument_name": f"{currency}-PERP"})
        if ticker and isinstance(ticker, dict):
            idx_price = ticker.get("index_price") or ticker.get("mark_price") or ticker.get("I")
            if idx_price is not None:
                return self._safe_float(idx_price)
        return None

    def get_all_active_expiries_with_dte(self, currency: str = "SOL") -> List[Tuple[str, float]]:
        """
        Pobiera z gieldy pelen harmonogram i wylicza dokladne DTE dla kazdej serii.
        """
        instruments = self._send_post("get_instruments", {
            "currency": currency,
            "instrument_type": "option",
            "expired": False
        })
        
        if not instruments or not isinstance(instruments, list):
            return []

        current_time_sec = time.time()
        expiry_map = {}

        for inst in instruments:
            if not isinstance(inst, dict):
                continue
            name = inst.get("instrument_name", "")
            parts = name.split("-")
            if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
                exp_str = parts[1]
                if exp_str not in expiry_map:
                    opt_details = inst.get("option_details", {})
                    exp_ts = opt_details.get("expiry") if isinstance(opt_details, dict) else None
                    if not exp_ts:
                        exp_ts = inst.get("scheduled_deactivation", 0)
                    
                    exp_ts_sec = self._safe_float(exp_ts)
                    dte = (exp_ts_sec - current_time_sec) / 86400.0
                    if dte > 0:
                        expiry_map[exp_str] = round(dte, 2)

        return sorted(expiry_map.items(), key=lambda x: x[1])

    def get_target_expiries(self, currency: str = "SOL", min_dte: float = 7.0, max_dte: float = 35.0) -> List[str]:
        """
        Filtruje serie, wybierajac tylko te z okna inwestycyjnego [min_dte, max_dte].
        """
        all_expiries = self.get_all_active_expiries_with_dte(currency=currency)
        return [exp_str for exp_str, dte in all_expiries if min_dte <= dte <= max_dte]

    def _normalize_tickers(self, raw_response: Any) -> List[Dict[str, Any]]:
        """
        Konwertuje odpowiedz API do listy slownikow z przypisana nazwa instrumentu.
        """
        if not raw_response:
            return []

        data = raw_response
        if isinstance(data, dict):
            if "tickers" in data:
                data = data["tickers"]
            elif "result" in data:
                data = data["result"]

        normalized = []

        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    if "instrument_name" not in val:
                        val["instrument_name"] = str(key)
                    normalized.append(val)
                elif isinstance(val, list):
                    for sub_val in val:
                        if isinstance(sub_val, dict):
                            normalized.append(sub_val)
        elif isinstance(data, list):
            for elem in data:
                if isinstance(elem, dict):
                    normalized.append(elem)

        return normalized

    def get_tickers_for_expiry(self, expiry_date: str, currency: str = "SOL") -> List[Dict[str, Any]]:
        """
        Pobiera znormalizowane tickery dla konkretnej daty YYYYMMDD.
        """
        res = self._send_post("get_tickers", {
            "currency": currency,
            "instrument_type": "option",
            "expiry_date": expiry_date
        })
        return self._normalize_tickers(res)

    def get_clean_options_chain(
        self,
        currency: str = "SOL",
        min_dte: float = 7.0,
        max_dte: float = 35.0,
        specific_expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Glowna metoda analityczna. Zwraca standaryzowana tabele opcji ze wszystkimi grekami.
        """
        spot_price = self.get_spot_price(currency=currency)
        all_expiries = self.get_all_active_expiries_with_dte(currency=currency)
        exp_dte_map = dict(all_expiries)

        if specific_expiry:
            target_expiries = [specific_expiry]
        else:
            target_expiries = [exp_str for exp_str, dte in all_expiries if min_dte <= dte <= max_dte]

        if not target_expiries:
            print(f"[INFO] Brak serii w zadanym oknie DTE ({min_dte} - {max_dte} dni) dla {currency}.")
            return pd.DataFrame()

        records = []

        for exp in target_expiries:
            dte = exp_dte_map.get(exp, 0.0)
            t_annual = dte / 365.25
            
            tickers = self.get_tickers_for_expiry(expiry_date=exp, currency=currency)

            for item in tickers:
                if not isinstance(item, dict):
                    continue

                instrument_name = item.get("instrument_name", "")
                parts = str(instrument_name).split("-")
                if len(parts) < 4:
                    continue

                try:
                    strike = self._safe_float(parts[2])
                    opt_type = "CALL" if parts[3].upper() == "C" else "PUT"
                except (ValueError, IndexError):
                    continue

                # Sub-obiekty telemetrii
                pricing = item.get("option_pricing")
                if not isinstance(pricing, dict):
                    pricing = {}

                stats = item.get("stats")
                if not isinstance(stats, dict):
                    stats = {}

                # Mapowanie cen: pelne nazwy ORAZ skroty Derive ('b', 'a', 'M', 'I')
                bid_price = self._safe_float(item.get("b") or item.get("best_bid_price"))
                ask_price = self._safe_float(item.get("a") or item.get("best_ask_price"))
                mark_price = self._safe_float(item.get("M") or pricing.get("m") or item.get("mark_price"))

                # Mapowanie grek i zmiennosci z sub-obiektu option_pricing ('d', 'g', 't', 'v', 'i')
                delta = self._safe_float(pricing.get("d") or pricing.get("delta") or item.get("delta"))
                gamma = self._safe_float(pricing.get("g") or pricing.get("gamma") or item.get("gamma"))
                theta = self._safe_float(pricing.get("t") or pricing.get("theta") or item.get("theta"))
                vega = self._safe_float(pricing.get("v") or pricing.get("vega") or item.get("vega"))
                iv = self._safe_float(pricing.get("i") or pricing.get("iv") or item.get("mark_iv"))
                
                # Otwarte pozycje z pod-obiektu stats
                oi = self._safe_float(stats.get("oi") or item.get("open_interest"))

                records.append({
                    "instrument_name": instrument_name,
                    "currency": currency,
                    "expiry_str": exp,
                    "dte": dte,
                    "T": round(t_annual, 6),
                    "strike": strike,
                    "option_type": opt_type,
                    "spot_price": spot_price,
                    "bid": bid_price,
                    "ask": ask_price,
                    "mark": mark_price,
                    "iv": round(iv, 4),
                    "delta": round(delta, 4),
                    "gamma": round(gamma, 6),
                    "theta": round(theta, 4),
                    "vega": round(vega, 4),
                    "oi": oi
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by=["dte", "strike", "option_type"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df


if __name__ == "__main__":
    print("Inicjalizacja zintegrowanego klienta Derive...")
    client = DeriveClient()

    spot = client.get_spot_price("SOL")
    print(f"[OK] Biezacy kurs referencyjny SOL: {spot} USD")

    MIN_DTE = 7.0
    MAX_DTE = 35.0

    df_chain = client.get_clean_options_chain("SOL", min_dte=MIN_DTE, max_dte=MAX_DTE)
    print(f"[SUKCES] Pobrano i znormalizowano w pamieci RAM {len(df_chain)} opcji!")

    if not df_chain.empty:
        first_expiry = df_chain["expiry_str"].iloc[0]
        print(f"\nPodglad opcji w okolicach ATM (Strike 115 - 125 USD) dla serii {first_expiry}:")
        atm_view = df_chain[
            (df_chain["expiry_str"] == first_expiry) &
            (df_chain["strike"] >= 115) & 
            (df_chain["strike"] <= 125)
        ]
        print(atm_view[["instrument_name", "strike", "option_type", "bid", "ask", "mark", "iv", "delta", "gamma", "theta", "oi"]])