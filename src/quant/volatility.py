import os
import sys
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class VolatilityEngine:
    """
    Rygorystyczny, zoptymalizowany numerycznie silnik estymacji zmienności zrealizowanej (RV),
    dynamicznej klasyfikacji reżimów rynkowych oraz adaptacyjnej premii za ryzyko zmienności (VRP).
    Działa w pamięci RAM w oparciu o wektoryzację NumPy, zachowując ścisły standard krypto ACT/365.0.
    """

    # Standard rynkowy krypto ACT/365.0 (24h na dobę, 365 dni w roku)
    ANNUALIZATION_DAILY: float = 365.0
    ANNUALIZATION_HOURLY: float = 365.0 * 24.0  # Dokładnie 8760.0 godzin w roku

    PI_OVER_2: float = math.pi / 2.0                       # ~1.57079633 (stała Bipower Variation)
    GARMAN_KLASS_CONST: float = 2.0 * math.log(2.0) - 1.0  # ~0.38629436
    PARKINSON_CONST: float = 1.0 / (4.0 * math.log(2.0))   # ~0.36067376

    @classmethod
    def _sanitize_ohlc(cls, df: pd.DataFrame, req_cols: set) -> pd.DataFrame:
        """
        Odrzuca świece uszkodzone, niekompletne lub niefizyczne (ceny <= 0, high < low).
        """
        if df.empty or not req_cols.issubset(df.columns):
            return pd.DataFrame()

        cond = pd.Series(True, index=df.index)
        for col in req_cols:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            cond = cond & (numeric_col > 0.0)

        if "high" in req_cols and "low" in req_cols:
            cond = cond & (pd.to_numeric(df["high"], errors="coerce") >= pd.to_numeric(df["low"], errors="coerce"))

        return df[cond].copy()

    @classmethod
    def calculate_log_returns(cls, df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        """
        Oblicza logarytmiczne stopy zwrotu po odrzuceniu cen niedodatnich.
        """
        if price_col not in df.columns or len(df) < 2:
            return pd.Series(dtype=float)

        prices = pd.to_numeric(df[price_col], errors="coerce")
        valid_prices = prices[prices > 0.0]
        if len(valid_prices) < 2:
            return pd.Series(dtype=float)

        return np.log(valid_prices / valid_prices.shift(1)).dropna()

    @classmethod
    def close_to_close(
        cls,
        df: pd.DataFrame,
        annualization_factor: float = ANNUALIZATION_HOURLY,
        ddof: int = 1
    ) -> float:
        """
        Klasyczny estymator zmienności zrealizowanej Close-to-Close.
        """
        log_ret = cls.calculate_log_returns(df, price_col="close")
        if len(log_ret) < 2:
            return 0.0
        return float(math.sqrt(max(log_ret.var(ddof=ddof), 0.0) * annualization_factor))

    @classmethod
    def parkinson(
        cls,
        df: pd.DataFrame,
        annualization_factor: float = ANNUALIZATION_HOURLY
    ) -> float:
        """
        Estymator Parkinsona oparty na rozpiętości High-Low.
        Odporny na luki danych i zera w arkuszu.
        """
        df_clean = cls._sanitize_ohlc(df, {"high", "low"})
        if len(df_clean) < 1:
            return 0.0

        high = df_clean["high"].astype(float).values
        low = df_clean["low"].astype(float).values

        ratio = high / low
        ratio = np.maximum(ratio, 1.0)
        log_hl_sq = np.log(ratio) ** 2.0
        variance = cls.PARKINSON_CONST * np.mean(log_hl_sq)

        return float(math.sqrt(max(variance, 0.0) * annualization_factor))

    @classmethod
    def garman_klass(
        cls,
        df: pd.DataFrame,
        annualization_factor: float = ANNUALIZATION_HOURLY
    ) -> float:
        """
        Estymator Garmana-Klassa (OHLC). Wykorzystuje pełną informację wewnątrzświecową.
        Zabezpieczony przed ujemną wariancją resztkową oraz błędami feedu.
        """
        df_clean = cls._sanitize_ohlc(df, {"open", "high", "low", "close"})
        if len(df_clean) < 1:
            return 0.0

        open_p = df_clean["open"].astype(float).values
        high_p = df_clean["high"].astype(float).values
        low_p = df_clean["low"].astype(float).values
        close_p = df_clean["close"].astype(float).values

        ratio_hl = np.maximum(high_p / low_p, 1.0)
        ratio_co = close_p / open_p

        term1 = 0.5 * (np.log(ratio_hl) ** 2.0)
        term2 = cls.GARMAN_KLASS_CONST * (np.log(ratio_co) ** 2.0)

        # Kluczowe zabezpieczenie: różnica na poziomie świecy nie może być ujemna
        diff = np.maximum(term1 - term2, 0.0)
        variance = np.mean(diff)

        return float(math.sqrt(max(variance, 0.0) * annualization_factor))

    @classmethod
    def jump_robust_bipower_variation(
        cls,
        df: pd.DataFrame,
        annualization_factor: float = ANNUALIZATION_HOURLY
    ) -> float:
        """
        Odporny na skoki cenowe estymator Bipower Variation (Barndorff-Nielsen & Shephard).
        Filtruje zniekształcenia wywołane jednorazowymi flash-crashami lub pompami.
        """
        log_ret = cls.calculate_log_returns(df, price_col="close").values
        n = len(log_ret)
        if n < 2:
            return 0.0

        abs_ret = np.abs(log_ret)
        bipower_sum = np.sum(abs_ret[1:] * abs_ret[:-1])
        variance = cls.PI_OVER_2 * (bipower_sum / (n - 1))

        return float(math.sqrt(max(variance, 0.0) * annualization_factor))

    @classmethod
    def rolling_garman_klass(
        cls,
        df: pd.DataFrame,
        window: int = 24,
        annualization_factor: float = ANNUALIZATION_HOURLY
    ) -> pd.Series:
        """
        Krocząca zmienność zrealizowana Garmana-Klassa w zadanym oknie czasowym.
        """
        df_clean = cls._sanitize_ohlc(df, {"open", "high", "low", "close"})
        if len(df_clean) < window:
            return pd.Series(dtype=float)

        open_p = df_clean["open"].astype(float)
        high_p = df_clean["high"].astype(float)
        low_p = df_clean["low"].astype(float)
        close_p = df_clean["close"].astype(float)

        ratio_hl = np.maximum(high_p / low_p, 1.0)
        ratio_co = close_p / open_p

        term1 = 0.5 * (np.log(ratio_hl) ** 2.0)
        term2 = cls.GARMAN_KLASS_CONST * (np.log(ratio_co) ** 2.0)
        diff = np.maximum(term1 - term2, 0.0)

        rolling_var = diff.rolling(window=window).mean()
        rolling_var = np.maximum(rolling_var, 0.0)

        rolling_vol = np.sqrt(rolling_var * annualization_factor)
        return rolling_vol.dropna()

    @classmethod
    def classify_volatility_regime(
        cls,
        rolling_rv_series: pd.Series,
        current_rv: float,
        q_low: float = 0.25,
        q_high: float = 0.75
    ) -> Dict[str, Any]:
        """
        Klasyfikuje reżim zmienności rynkowej na podstawie rozkładu empirycznego.
        Ściśle ujednolica schemat zwracanych kluczy słownika (brak ryzyka KeyError w downstreamie).
        """
        # Standaryzowany kontrakt słownika dla stanu niepełnych danych
        if len(rolling_rv_series) < 10:
            return {
                "regime": "INSUFFICIENT_DATA",
                "current_rv": round(current_rv, 4),
                "percentile_rank": 0.50,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0
            }

        q25 = float(np.percentile(rolling_rv_series, q_low * 100.0))
        median = float(np.median(rolling_rv_series))
        q75 = float(np.percentile(rolling_rv_series, q_high * 100.0))

        percentile_rank = float(np.mean(rolling_rv_series <= current_rv))

        if current_rv < q25:
            regime = "LOW_VOLATILITY"
        elif current_rv > q75:
            regime = "HIGH_VOLATILITY"
        else:
            regime = "NORMAL_VOLATILITY"

        return {
            "regime": regime,
            "current_rv": round(current_rv, 4),
            "percentile_rank": round(percentile_rank, 4),
            "q25": round(q25, 4),
            "median": round(median, 4),
            "q75": round(q75, 4)
        }

    @classmethod
    def calculate_adaptive_vrp(
        cls,
        iv: float,
        rv: float,
        regime_info: Optional[Dict[str, Any]] = None,
        rolling_vrp_series: Optional[pd.Series] = None,
        dte: Optional[float] = None,
        rv_horizon_days: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Oblicza adaptacyjną premię za ryzyko zmienności (VRP = IV - RV).
        Zapewnia spójność progów decyzyjnych z reżimem rynkowym oraz monitoruje
        dopasowanie horyzontu czasowego (term structure horizon consistency).
        """
        vrp_abs = iv - rv
        vrp_rel = (vrp_abs / rv) if rv > 0.0 else 0.0

        z_score = None
        vrp_percentile = None

        if rolling_vrp_series is not None and len(rolling_vrp_series) >= 10:
            mean_vrp = float(np.mean(rolling_vrp_series))
            std_vrp = float(np.std(rolling_vrp_series, ddof=1))
            if std_vrp > 1e-8:
                z_score = round(float((vrp_abs - mean_vrp) / std_vrp), 2)
            vrp_percentile = round(float(np.mean(rolling_vrp_series <= vrp_abs)), 4)

        regime = regime_info.get("regime", "NORMAL_VOLATILITY") if regime_info else "NORMAL_VOLATILITY"

        # Adaptacja progów Z-Score oraz progów relatywnych do zidentyfikowanego reżimu
        if regime == "LOW_VOLATILITY":
            z_sell_threshold, z_buy_threshold = 1.2, -1.8
            rich_rel_threshold, cheap_rel_threshold = 0.20, 0.00
        elif regime == "HIGH_VOLATILITY":
            z_sell_threshold, z_buy_threshold = 1.8, -1.2
            rich_rel_threshold, cheap_rel_threshold = 0.15, -0.05
        else:
            z_sell_threshold, z_buy_threshold = 1.5, -1.5
            rich_rel_threshold, cheap_rel_threshold = 0.18, 0.00

        # Weryfikacja spójności horyzontów czasowych
        term_structure_mismatch = False
        if dte is not None and rv_horizon_days is not None:
            # Flaga ostrzegawcza, jeśli horyzonty różnią się ponad 2-krotnie
            if max(dte, rv_horizon_days) / max(min(dte, rv_horizon_days), 1e-4) > 2.0:
                term_structure_mismatch = True

        if z_score is not None:
            if z_score >= z_sell_threshold:
                signal = "OVERPRICED_SELL_PREMIUM"
            elif z_score <= z_buy_threshold:
                signal = "UNDERPRICED_BUY_CONVEXITY"
            else:
                signal = "FAIRLY_PRICED_DIRECTIONAL"
        else:
            if vrp_rel >= rich_rel_threshold:
                signal = "OVERPRICED_SELL_PREMIUM"
            elif vrp_rel <= cheap_rel_threshold:
                signal = "UNDERPRICED_BUY_CONVEXITY"
            else:
                signal = "FAIRLY_PRICED_DIRECTIONAL"

        return {
            "iv": round(iv, 4),
            "rv": round(rv, 4),
            "vrp_absolute": round(vrp_abs, 4),
            "vrp_relative": round(vrp_rel, 4),
            "vrp_relative_pct": round(vrp_rel * 100.0, 2),
            "regime_context": regime,
            "z_score": z_score,
            "vrp_percentile": vrp_percentile,
            "term_structure_mismatch": term_structure_mismatch,
            "signal": signal
        }

    @classmethod
    def calculate_funding_drift_adjustment(
        cls,
        spot: float,
        funding_rate_8h: float,
        T: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Kalkulacja dryfu wyceny wywołanego stopą finansowania kontraktów perpetualnych.
        Baza ujednolicona do 365.0 dni w roku (ACT/365.0).
        """
        q_perp = funding_rate_8h * 3.0 * cls.ANNUALIZATION_DAILY
        f_traditional = spot * math.exp(risk_free_rate * T)
        f_crypto_adj = spot * math.exp((risk_free_rate - q_perp) * T)
        drift_diff = f_crypto_adj - f_traditional

        return {
            "spot": spot,
            "q_perp_annualized": round(q_perp, 4),
            "forward_traditional": round(f_traditional, 4),
            "forward_crypto_adjusted": round(f_crypto_adj, 4),
            "drift_impact_usd": round(drift_diff, 4)
        }


if __name__ == "__main__":
    from src.data_feed.market_spot import MarketSpotClient
    from src.data_feed.derive_client import DeriveClient

    print("==================================================")
    print("TEST MODUŁU VOLATILITYENGINE NA ŻYWYM RYNKU (RAM)")
    print("==================================================")

    spot_client = MarketSpotClient()
    derive_client = DeriveClient()

    print("1. Pobieranie świec spotowych z API...")
    df_candles = spot_client.get_clean_spot_candles(symbol="SOLUSDC", interval="1h", limit=168, save_csv=False)
    print(f"[OK] Pobrano {len(df_candles)} świec bezpośrednio do RAM.")

    print("\n2. Pobieranie łańcucha opcji z Derive...")
    df_options = derive_client.get_clean_options_chain("SOL", min_dte=7.0, max_dte=35.0)
    print(f"[OK] Pobrano {len(df_options)} opcji bezpośrednio do RAM.")

    spot_price = df_candles.iloc[-1]["close"]
    first_exp = df_options["expiry_str"].iloc[0]
    exp_df = df_options[df_options["expiry_str"] == first_exp]

    calls = exp_df[exp_df["option_type"] == "CALL"].sort_values(by="strike")
    atm_idx = (calls["strike"] - spot_price).abs().idxmin()
    atm_call = calls.loc[atm_idx]

    live_iv = float(atm_call["iv"])
    live_dte = float(atm_call["dte"])
    atm_strike = float(atm_call["strike"])
    print(f"[OK] Seria: {first_exp} (DTE: {live_dte}d) | Strike ATM: {atm_strike} USD | IV: {live_iv * 100:.2f}%")

    print("\n3. Estymatory zmienności (Baza ACT/365.0):")
    c2c_vol = VolatilityEngine.close_to_close(df_candles)
    park_vol = VolatilityEngine.parkinson(df_candles)
    gk_vol = VolatilityEngine.garman_klass(df_candles)
    bv_vol = VolatilityEngine.jump_robust_bipower_variation(df_candles)

    print(f"  - Close-to-Close RV:                {c2c_vol * 100:.2f}%")
    print(f"  - Parkinson RV (High-Low):          {park_vol * 100:.2f}%")
    print(f"  - Garman-Klass RV (OHLC):           {gk_vol * 100:.2f}%")
    print(f"  - Jump-Robust Bipower (Bez skoków): {bv_vol * 100:.2f}%")

    rolling_gk = VolatilityEngine.rolling_garman_klass(df_candles, window=24)
    regime_res = VolatilityEngine.classify_volatility_regime(rolling_gk, current_rv=rolling_gk.iloc[-1])
    print(f"\n4. Diagnostyka reżimu rynkowego:")
    print(f"  - Zidentyfikowany reżim:           {regime_res['regime']}")
    print(f"  - Bieżące RV (24h):                {regime_res['current_rv'] * 100:.2f}%")
    print(f"  - Progi kwantylowe:                q25={regime_res['q25']:.4f}, median={regime_res['median']:.4f}, q75={regime_res['q75']:.4f}")

    vrp_res = VolatilityEngine.calculate_adaptive_vrp(
        iv=live_iv,
        rv=rolling_gk.iloc[-1],
        regime_info=regime_res,
        dte=live_dte,
        rv_horizon_days=1.0  # Okno 24h
    )
    print(f"\n5. Adaptacyjny sygnał VRP:")
    print(f"  - VRP Względny:                    {vrp_res['vrp_relative_pct']:+.2f}%")
    print(f"  - Sygnał:                          {vrp_res['signal']}")
    print(f"  - Ostrzeżenie o horyzoncie tenoru: {vrp_res['term_structure_mismatch']}")