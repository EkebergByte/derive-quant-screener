import os
import sys
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class PriceProjectionEngine:
    """
    Rygorystyczny silnik projekcji prawdopodobienstwa i stozkow cenowych.
    Wylicza przedzialy ufnosci w czasie oraz analityczne prawdopodobienstwo
    dotkniecia poziomow docelowych (First-Passage Time / Probability of Touching).
    Dziala w mierze fizycznej P z prawidlowa korekta Ito dla geometrycznego ruchu Browna (GBM)
    oraz obsluguje warunki brzegowe dla pozycji juz znajdujacych sie w zysku (ITM).
    """

    DAYS_PER_YEAR: float = 365.0

    @staticmethod
    def _clean_numeric(val: Any, default: float = 0.0) -> float:
        """
        Bezpieczna konwersja do float odporna na napisy typu '118 USD' lub '+7.87 USD'.
        """
        if val is None or pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            cleaned = str(val).replace("USD", "").replace("+", "").replace("%", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return default

    @classmethod
    def calculate_first_passage_touch(
        cls,
        spot: float,
        barrier: float,
        time_years: float,
        volatility: float,
        drift: float = 0.0
    ) -> float:
        """
        Scisle analityczne prawdopodobienstwo dotkniecia bariery cenowej
        w dowolnym momencie w przedziale czasu [0, time_years].
        Zabezpieczone obustronnie przed przepelnieniem wykladniczym.
        """
        if time_years <= 1e-6 or volatility <= 1e-4 or spot <= 0.0 or barrier <= 0.0:
            return 1.0 if (spot >= barrier if barrier >= spot else spot <= barrier) else 0.0

        if math.isclose(spot, barrier, rel_tol=1e-5):
            return 1.0

        # Dryf procesu logarytmicznego: nu = mu - 0.5 * sigma^2
        nu = drift - 0.5 * (volatility ** 2.0)
        denom = volatility * math.sqrt(time_years)

        # Bariera gorna (Upper Barrier: H > Spot)
        if barrier > spot:
            m = math.log(barrier / spot)
            term1 = norm.cdf((-m + nu * time_years) / denom)
            expo = (2.0 * nu * m) / (volatility ** 2.0)

            # Obustronne zabezpieczenie numeryczne
            if abs(expo) > 700.0:
                term2 = 0.0
            else:
                term2 = math.exp(expo) * norm.cdf((-m - nu * time_years) / denom)

            pot = term1 + term2
            return float(min(1.0, max(0.0, pot)))

        # Bariera dolna (Lower Barrier: L < Spot)
        else:
            m = math.log(barrier / spot)  # wartosc ujemna
            term1 = norm.cdf((m - nu * time_years) / denom)
            expo = (2.0 * nu * m) / (volatility ** 2.0)

            # Obustronne zabezpieczenie numeryczne
            if abs(expo) > 700.0:
                term2 = 0.0
            else:
                term2 = math.exp(expo) * norm.cdf((m + nu * time_years) / denom)

            pot = term1 + term2
            return float(min(1.0, max(0.0, pot)))

    @classmethod
    def calculate_expiring_beyond(
        cls,
        spot: float,
        target: float,
        time_years: float,
        volatility: float,
        drift: float = 0.0,
        is_upper: bool = True
    ) -> float:
        """
        Prawdopodobienstwo zakonczenia wygasniecia za celem cenowym w momencie T.
        """
        if time_years <= 1e-6 or volatility <= 1e-4:
            return 1.0 if (spot >= target if is_upper else spot <= target) else 0.0

        nu = drift - 0.5 * (volatility ** 2.0)
        denom = volatility * math.sqrt(time_years)

        if is_upper:
            d2 = (math.log(spot / target) + nu * time_years) / denom
            return float(norm.cdf(d2))
        else:
            d2 = (math.log(target / spot) - nu * time_years) / denom
            return float(norm.cdf(d2))

    @classmethod
    def build_confidence_cones(
        cls,
        spot: float,
        dte_days: float,
        volatility: float,
        drift: float = 0.0
    ) -> pd.DataFrame:
        """
        Generuje korytarze stozka zmiennosci dla 50%, 68.3% (1-sigma) i 95.4% (2-sigmy)
        wycentrowane scisle na medianie geometrycznej (korekta Ito).
        """
        checkpoints = []
        for d in [7.0, 14.0, 21.0, dte_days]:
            if d <= dte_days and d not in checkpoints:
                checkpoints.append(d)
        if dte_days not in checkpoints:
            checkpoints.append(dte_days)
        checkpoints = sorted(checkpoints)

        rows = []
        for days in checkpoints:
            t_yrs = days / cls.DAYS_PER_YEAR

            # Srodek rozkladu log-normalnego (mediana) z korekta Ito: -0.5 * sigma^2
            nu = drift - 0.5 * (volatility ** 2.0)
            median_forward = spot * math.exp(nu * t_yrs)

            # Wartosc oczekiwana (srednia arytmetyczna forwardu)
            mean_forward = spot * math.exp(drift * t_yrs)

            std_dev = volatility * math.sqrt(t_yrs)

            # Poziom 50% (Z = 0.67449)
            low_50 = median_forward * math.exp(-0.67449 * std_dev)
            high_50 = median_forward * math.exp(0.67449 * std_dev)

            # Poziom 68.27% (1-sigma, Z = 1.0)
            low_68 = median_forward * math.exp(-1.0 * std_dev)
            high_68 = median_forward * math.exp(1.0 * std_dev)

            # Poziom 95.45% (2-sigma, Z = 1.95996)
            low_95 = median_forward * math.exp(-1.95996 * std_dev)
            high_95 = median_forward * math.exp(1.95996 * std_dev)

            rows.append({
                "HORYZONT": f"{days:.1f} dni",
                "MEDIANA (ITO)": f"{median_forward:.2f} USD",
                "SREDNIA FWD": f"{mean_forward:.2f} USD",
                "PASMO 50% (DOL - GORA)": f"{low_50:.2f} - {high_50:.2f} USD",
                "PASMO 68% (1-SIGMA)": f"{low_68:.2f} - {high_68:.2f} USD",
                "PASMO 95% (2-SIGMY)": f"{low_95:.2f} - {high_95:.2f} USD"
            })

        return pd.DataFrame(rows)

    @classmethod
    def evaluate_spread_milestones(
        cls,
        spot: float,
        dte_days: float,
        volatility: float,
        drift: float = 0.0,
        strategy_row: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Wylicza matryce celow dla wybranego spreadu z pelna normalizacja kontraktu,
        wlasciwym kierunkiem oraz obsluga pozycji juz znajdujacych sie w zysku (ITM).
        """
        if strategy_row is None:
            strategy_row = {}

        t_total = dte_days / cls.DAYS_PER_YEAR

        # 1. Defensywna normalizacja kontraktu strategii
        strat_raw = str(strategy_row.get("STRATEGY") or strategy_row.get("STRAT") or "").strip().upper()
        if "BULL" in strat_raw:
            is_bull = True
        elif "BEAR" in strat_raw:
            is_bull = False
        else:
            raise ValueError(f"Nieznany typ strategii: {strat_raw}. Oczekiwano prefiksu BULL lub BEAR.")

        k_long = cls._clean_numeric(strategy_row.get("LONG_K"), spot)
        k_short = cls._clean_numeric(strategy_row.get("SHORT_K"), spot)
        cost = abs(cls._clean_numeric(strategy_row.get("NET_COST"), 0.0))
        max_win = cls._clean_numeric(strategy_row.get("MAX_WIN"), 0.0)

        # 2. Bezpieczna obsluga progu rentownosci z fallbackiem
        raw_be = strategy_row.get("BREAKEVEN")
        cleaned_be = cls._clean_numeric(raw_be, default=-1.0)
        if cleaned_be > 0.0:
            breakeven = cleaned_be
        else:
            breakeven = (k_long + cost) if is_bull else (k_long - cost)

        # 3. Definicja 3 kluczowych poziomow kierunkowych
        if is_bull:
            t1_price = breakeven
            t1_desc = "Prog rentownosci (Breakeven - wyjscie na zero)"

            t2_price = k_long + cost + 0.5 * max_win
            t2_desc = "Cel 50% maksymalnego zysku (Wczesny Take Profit)"

            t3_price = k_short
            t3_desc = "Cel 100% pelnego zysku (Short Strike)"
        else:
            t1_price = breakeven
            t1_desc = "Prog rentownosci (Breakeven - wyjscie na zero)"

            t2_price = k_long - cost - 0.5 * max_win
            t2_desc = "Cel 50% maksymalnego zysku (Wczesny Take Profit)"

            t3_price = k_short
            t3_desc = "Cel 100% pelnego zysku (Short Strike)"

        targets = [
            (t1_desc, t1_price),
            (t2_desc, t2_price),
            (t3_desc, t3_price)
        ]

        rows = []
        for desc, p_target in targets:
            # OPTYMALIZACJA: Jesli cel jest juz przekroczony na starcie (ITM), PoT = 1.0 (100%)
            is_already_reached = (spot >= p_target) if is_bull else (spot <= p_target)

            if is_already_reached:
                pot = 1.0
            else:
                pot = cls.calculate_first_passage_touch(
                    spot=spot,
                    barrier=p_target,
                    time_years=t_total,
                    volatility=volatility,
                    drift=drift
                )

            # Prawdopodobienstwo zakonczenia na wygasnieciu za celem (PoE)
            poe = cls.calculate_expiring_beyond(
                spot=spot,
                target=p_target,
                time_years=t_total,
                volatility=volatility,
                drift=drift,
                is_upper=is_bull
            )

            # Spojnosc stochastyczna: PoT nie moze byc mniejszy niz PoE
            pot = max(pot, poe)

            diff_usd = p_target - spot
            diff_pct = (diff_usd / spot) * 100.0

            rows.append({
                "KAMIEŃ MILOWY": desc,
                "KURS CELU": f"{p_target:.2f} USD",
                "DYSTANS OD SPOT": f"{diff_usd:+.2f} USD ({diff_pct:+.1f}%)",
                "SZANSA DOTKNIECIA (PoT)": f"{pot * 100.0:.1f}%",
                "SZANSA NA WYGASNIECIU (PoE)": f"{poe * 100.0:.1f}%"
            })

        return pd.DataFrame(rows)

    @classmethod
    def display_projection_report(
        cls,
        spot: float,
        volatility: float,
        drift: float = 0.0,
        strategy_row: Dict[str, Any] = None
    ):
        if strategy_row is None:
            strategy_row = {}

        strat = str(strategy_row.get("STRATEGY") or strategy_row.get("STRAT") or "STRATEGIA").upper()
        exp = strategy_row.get("EXPIRY", "")
        dte = cls._clean_numeric(strategy_row.get("DTE"), 0.0)
        k_long = strategy_row.get("LONG_K", "N/A")
        k_short = strategy_row.get("SHORT_K", "N/A")
        cost = strategy_row.get("NET_COST", "N/A")
        max_win = strategy_row.get("MAX_WIN", "N/A")
        rr = strategy_row.get("RR_RATIO", "N/A")

        print("\n========================================================================================================================")
        print(f"                       RAPORT PROJEKCJI PRAWDOPODOBIENSTWA DLA: {strat} {k_long} / {k_short} ({exp})")
        print("========================================================================================================================")
        print(f"Biezacy kurs SOL:  {spot:.2f} USD | Czas do wygasniecia (DTE): {dte:.1f} dni")
        print(f"Uzyta zmiennosc:   {volatility * 100.0:.2f}% | Roczny dryf projekcji (miara P): {drift * 100.0:.2f}%")
        print(f"Parametry pozycji: Koszt: {cost} | Max Zysk: {max_win} | Stosunek R/R: {rr}")
        print("------------------------------------------------------------------------------------------------------------------------")

        # 1. Tabela stozka zmiennosci wycentrowana na medianie Ito
        print("\n[1] STOZEK ZMIENNOSCI (KORYTARZE PRAWDOPODOBIENSTWA CENY SOL W CZASIE):")
        df_cones = cls.build_confidence_cones(spot=spot, dte_days=dte, volatility=volatility, drift=drift)
        if HAS_TABULATE:
            print(tabulate(df_cones, headers="keys", tablefmt="fancy_grid", showindex=False))
        else:
            print(df_cones.to_string(index=False))

        # 2. Tabela szansy realizacji celow
        print("\n[2] MATRYCA REALIZACJI CELOW POZYCJI (DOTKNIECIE W TRAKCIE VS ZAKONCZENIE NA WYGASNIECIU):")
        df_targets = cls.evaluate_spread_milestones(
            spot=spot, dte_days=dte, volatility=volatility, drift=drift, strategy_row=strategy_row
        )
        if HAS_TABULATE:
            print(tabulate(df_targets, headers="keys", tablefmt="fancy_grid", showindex=False))
        else:
            print(df_targets.to_string(index=False))

        print("========================================================================================================================\n")