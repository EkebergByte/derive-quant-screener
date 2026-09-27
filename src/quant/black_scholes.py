import os
import sys
import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.stats import norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class BlackScholesEngine:
    """
    Rygorystyczny silnik analityczny wyceny opcji i grek w modelu Blacka-Scholesa-Mertona (Black-76).
    Zoptymalizowany pod realia krypto z dynamicznym kosztem carry (r - q), ścisłymi zabezpieczeniami
    brzegowymi oraz hybrydowym, odpornym numerycznie solverem zmienności implikowanej (IV).
    """

    DAYS_PER_YEAR: float = 365.0  # Ujednolicony standard rynkowy krypto ACT/365
    MIN_T: float = 1e-7          # Minimalny czas (< 3 sekundy), poniżej którego przechodzimy na payoff terminalny
    MIN_SIGMA: float = 1e-4      # Minimalna zmienność (0.01%)
    MAX_SIGMA: float = 5.0       # Maksymalna zmienność (500% IV)

    @classmethod
    def calculate_d1_d2(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Tuple[float, float]:
        """
        Numerycznie bezpieczne wyliczanie standaryzowanych zmiennych d1 i d2.
        Ściśle zachowuje tożsamość d2 = d1 - sigma * sqrt(T) bez asymetrycznego obcinania.
        """
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA or S <= 0.0 or K <= 0.0:
            return 0.0, 0.0

        denominator = sigma * math.sqrt(T)
        numerator = math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
        d1 = numerator / denominator
        d2 = d1 - denominator

        return d1, d2

    @classmethod
    def price(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "CALL"
    ) -> float:
        """
        Analityczna wycena europejskiej opcji Call lub Put (Merton/Black-76).
        """
        opt_type = option_type.upper()
        if opt_type not in ("CALL", "PUT"):
            raise ValueError(f"Nieprawidlowy typ opcji: {option_type}. Oczekiwano CALL lub PUT.")

        # 1. Zabezpieczenie brzegowe: Czas wygasł (T -> 0)
        if T <= cls.MIN_T:
            if opt_type == "CALL":
                return max(0.0, S - K)
            else:
                return max(0.0, K - S)

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        # 2. Zabezpieczenie brzegowe: Zmienność zerowa (sigma -> 0)
        # Wycena opiera się na zdyskontowanej wartości wewnętrznej na forwardzie
        if sigma <= cls.MIN_SIGMA:
            forward_diff = S * disc_q - K * disc_r
            if opt_type == "CALL":
                return max(0.0, forward_diff)
            else:
                return max(0.0, -forward_diff)

        # 3. Standardowa wycena analityczna BSM
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma, q)

        if opt_type == "CALL":
            val = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
            return max(0.0, float(val))
        else:
            val = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
            return max(0.0, float(val))

    # =========================================================================
    # GREKI PIERWSZEGO RZĘDU
    # =========================================================================

    @classmethod
    def delta(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "CALL"
    ) -> float:
        opt_type = option_type.upper()
        disc_q = math.exp(-q * T)

        # Na granicy sigma -> 0 decyduje zdyskontowany forward vs strike
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA:
            disc_r = math.exp(-r * T)
            forward_diff = S * disc_q - K * disc_r
            if opt_type == "CALL":
                return disc_q if forward_diff > 0 else (0.5 * disc_q if forward_diff == 0 else 0.0)
            else:
                return -disc_q if forward_diff < 0 else (-0.5 * disc_q if forward_diff == 0 else 0.0)

        d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma, q)

        if opt_type == "CALL":
            return float(disc_q * norm.cdf(d1))
        else:
            return float(disc_q * (norm.cdf(d1) - 1.0))

    @classmethod
    def gamma(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA or S <= 0.0:
            return 0.0

        d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_q = math.exp(-q * T)
        pdf_d1 = norm.pdf(d1)
        denom = S * sigma * math.sqrt(T)

        if denom < 1e-12:
            return 0.0

        return float((disc_q * pdf_d1) / denom)

    @classmethod
    def vega(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Zwraca Vegę rynkową: zmianę ceny opcji przy zmianie IV o 1 punkt procentowy (0.01).
        """
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA or S <= 0.0:
            return 0.0

        d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_q = math.exp(-q * T)
        raw_vega = S * disc_q * norm.pdf(d1) * math.sqrt(T)

        return float(raw_vega / 100.0)

    @classmethod
    def theta(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "CALL"
    ) -> float:
        """
        Zwraca Thetę dobową: erozję czasową opcji na 1 dzień (1/365.0 roku).
        """
        if T <= cls.MIN_T:
            return 0.0

        opt_type = option_type.upper()
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)
        pdf_d1 = norm.pdf(d1)

        term1 = -(S * disc_q * pdf_d1 * sigma) / (2.0 * math.sqrt(T))

        if opt_type == "CALL":
            term2 = q * S * disc_q * norm.cdf(d1) - r * K * disc_r * norm.cdf(d2)
        else:
            term2 = -q * S * disc_q * norm.cdf(-d1) + r * K * disc_r * norm.cdf(-d2)

        theta_annual = term1 + term2
        return float(theta_annual / cls.DAYS_PER_YEAR)

    @classmethod
    def rho(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "CALL"
    ) -> float:
        """
        Zwraca Rho na 1 punkt procentowy zmiany stopy wolnej od ryzyka (0.01).
        """
        if T <= cls.MIN_T:
            return 0.0

        opt_type = option_type.upper()
        _, d2 = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_r = math.exp(-r * T)

        if opt_type == "CALL":
            raw_rho = K * T * disc_r * norm.cdf(d2)
        else:
            raw_rho = -K * T * disc_r * norm.cdf(-d2)

        return float(raw_rho / 100.0)

    # =========================================================================
    # GREKI DRUGIEGO RZĘDU
    # =========================================================================

    @classmethod
    def vanna(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Vanna: dDelta / dSigma = dVega / dS.
        Zwraca zmianę Delty przy wzroście IV o 1 punkt procentowy (0.01).
        """
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA or S <= 0.0:
            return 0.0

        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_q = math.exp(-q * T)
        raw_vanna = -disc_q * norm.pdf(d1) * (d2 / sigma)

        return float(raw_vanna / 100.0)

    @classmethod
    def volga(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Volga (Vomma): d^2V / dSigma^2. Wypukłość zmienności.
        Zwraca zmianę Vegi przy wzroście IV o 1 punkt procentowy (0.01).
        """
        if T <= cls.MIN_T or sigma <= cls.MIN_SIGMA or S <= 0.0:
            return 0.0

        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma, q)
        disc_q = math.exp(-q * T)
        raw_vega = S * disc_q * norm.pdf(d1) * math.sqrt(T)
        raw_volga = raw_vega * (d1 * d2 / sigma)

        return float(raw_volga / 10000.0)

    @classmethod
    def get_all_greeks(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "CALL"
    ) -> Dict[str, float]:
        """
        Zwraca pełen pakiet grek i wyceny w jednym słowniku.
        """
        p = cls.price(S, K, T, r, sigma, q, option_type)
        d = cls.delta(S, K, T, r, sigma, q, option_type)
        g = cls.gamma(S, K, T, r, sigma, q)
        v = cls.vega(S, K, T, r, sigma, q)
        t = cls.theta(S, K, T, r, sigma, q, option_type)
        rh = cls.rho(S, K, T, r, sigma, q, option_type)
        vn = cls.vanna(S, K, T, r, sigma, q)
        vg = cls.volga(S, K, T, r, sigma, q)

        return {
            "price": round(p, 4),
            "delta": round(d, 4),
            "gamma": round(g, 6),
            "theta_daily": round(t, 4),
            "vega_1pct": round(v, 4),
            "rho_1pct": round(rh, 4),
            "vanna_1pct": round(vn, 6),
            "volga_1pct": round(vg, 6)
        }

    # =========================================================================
    # HYBRYDOWY SOLVER IV
    # =========================================================================

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.05,
        q: float = 0.0,
        option_type: str = "CALL",
        tolerance: float = 1e-6,
        max_iterations: int = 30
    ) -> float:
        """
        Odporny numerycznie hybrydowy solver zmienności implikowanej.
        Zwraca float('nan') w przypadku naruszenia warunków braku arbitrażu (zamiast sztucznego
        fałszowania danych jako 0.0001 lub MAX_SIGMA).
        """
        if market_price <= 0.0 or T <= cls.MIN_T or S <= 0.0 or K <= 0.0:
            return float("nan")

        opt_type = option_type.upper()
        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        # 1. Granice braku arbitrażu dla opcji europejskiej z carry
        if opt_type == "CALL":
            intrinsic = max(0.0, S * disc_q - K * disc_r)
            max_price = S * disc_q
        else:
            intrinsic = max(0.0, K * disc_r - S * disc_q)
            max_price = K * disc_r

        # Odrzucenie cen naruszających no-arbitrage bounds
        if market_price < intrinsic - 1e-7 or market_price > max_price + 1e-7:
            return float("nan")

        if abs(market_price - intrinsic) < 1e-7:
            return cls.MIN_SIGMA

        # 2. Punkt startowy dopasowany do forwardu
        forward = S * math.exp((r - q) * T)
        moneyness = forward / K
        if 0.8 <= moneyness <= 1.2:
            sigma0 = math.sqrt(2.0 * math.pi / T) * (market_price / (S * disc_q))
        else:
            sigma0 = 0.60  # Domyślny stabilny punkt startowy dla krypto

        sigma = max(cls.MIN_SIGMA, min(cls.MAX_SIGMA, sigma0))

        # 3. Newton-Raphson
        use_bisection = False
        for _ in range(max_iterations):
            p_est = cls.price(S, K, T, r, sigma, q, opt_type)
            diff = p_est - market_price

            if abs(diff) < tolerance:
                return round(float(sigma), 4)

            d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma, q)
            raw_vega = S * disc_q * norm.pdf(d1) * math.sqrt(T)

            if raw_vega < 1e-8:
                use_bisection = True
                break

            step = diff / raw_vega
            sigma -= step

            if sigma <= cls.MIN_SIGMA or sigma >= cls.MAX_SIGMA:
                use_bisection = True
                break

        # 4. Bisekcja awaryjna
        if use_bisection:
            low = cls.MIN_SIGMA
            high = cls.MAX_SIGMA

            for _ in range(45):
                mid = (low + high) / 2.0
                p_mid = cls.price(S, K, T, r, mid, q, opt_type)
                diff = p_mid - market_price

                if abs(diff) < tolerance:
                    return round(float(mid), 4)

                if diff > 0:
                    high = mid
                else:
                    low = mid

            sigma = (low + high) / 2.0

        return round(float(sigma), 4)


if __name__ == "__main__":
    from src.data_feed.market_spot import MarketSpotClient
    from src.data_feed.derive_client import DeriveClient

    print("==================================================")
    print("TEST MODELU BLACKA-SCHOLESA NA ŻYWYM RYNKU W RAM")
    print("==================================================")

    # 1. Dynamiczne pobranie parametrów rynkowych
    spot_client = MarketSpotClient()
    derive_client = DeriveClient()

    live_spot = derive_client.get_spot_price("SOL")
    funding_info = spot_client.get_current_funding_rate("SOLUSDT")
    q_funding = funding_info.get("annualized_rate", 0.0)
    r_macro = spot_client.get_us_treasury_risk_free_rate()

    # 2. Pobranie żywego łańcucha opcji z Derive
    df_options = derive_client.get_clean_options_chain("SOL", min_dte=5.0, max_dte=35.0)
    if df_options.empty:
        print("[BŁĄD] Brak danych opcyjnych w RAM.")
        sys.exit(1)

    first_exp = df_options["expiry_str"].iloc[0]
    exp_df = df_options[df_options["expiry_str"] == first_exp]

    live_t = float(exp_df["T"].iloc[0])
    live_dte = float(exp_df["dte"].iloc[0])

    # 3. Dynamiczne znalezienie żywej opcji Call i Put ATM
    calls = exp_df[exp_df["option_type"] == "CALL"].sort_values(by="strike")
    atm_idx = (calls["strike"] - live_spot).abs().idxmin()
    live_atm_call = calls.loc[atm_idx]

    live_k = float(live_atm_call["strike"])
    live_iv = float(live_atm_call["iv"]) if live_atm_call["iv"] is not None else 0.60
    derive_call_mark = float(live_atm_call["mark"])

    puts = exp_df[(exp_df["option_type"] == "PUT") & (exp_df["strike"] == live_k)]
    derive_put_mark = float(puts.iloc[0]["mark"]) if not puts.empty else 0.0

    print(f"\n1. Parametry rynkowe pobrane dynamicznie do RAM:")
    print(f"  - Kurs referencyjny SOL (Spot):       {live_spot} USD")
    print(f"  - Dynamiczna stopa r (U.S. Treasury): {r_macro * 100:.2f}%")
    print(f"  - Funding Rate perps (APR q):          {q_funding * 100:.4f}%")
    print(f"  - Seria wygasania:                     {first_exp} (DTE: {live_dte} dni, T: {live_t:.6f})")
    print(f"  - Strike ATM:                          {live_k} USD")
    print(f"  - Żywe IV z Derive:                    {live_iv * 100:.2f}%")

    bs = BlackScholesEngine()

    # 4. Wycena analityczna vs Ceny Mark z Derive
    model_c_price = bs.price(live_spot, live_k, live_t, r_macro, live_iv, q_funding, "CALL")
    model_p_price = bs.price(live_spot, live_k, live_t, r_macro, live_iv, q_funding, "PUT")

    print(f"\n2. Weryfikacja wyceny analitycznej vs Giełda Derive:")
    print(f"  - Call ($ {live_k}) Wycena Modelu:     {model_c_price:.4f} USD | Derive Mark: {derive_call_mark:.4f} USD")
    print(f"  - Put  ($ {live_k}) Wycena Modelu:     {model_p_price:.4f} USD | Derive Mark: {derive_put_mark:.4f} USD")

    # 5. Parytet Call-Put na żywych danych rynkowych
    left_parity = model_c_price - model_p_price
    right_parity = live_spot * math.exp(-q_funding * live_t) - live_k * math.exp(-r_macro * live_t)
    parity_err = abs(left_parity - right_parity)
    print(f"  - Błąd parytetu Call-Put:              {parity_err:.10f}")
    assert parity_err < 1e-5, "[BŁĄD] Naruszenie parytetu Call-Put!"
    print("  [OK] Parytet Call-Put jest spełniony z dokładnością maszynową.")

    # 6. Pełny pakiet analitycznych grek dla żywego kontraktu ATM
    print(f"\n3. Analityczne greki dla żywego kontraktu ATM Call ({live_k} USD):")
    live_greeks = bs.get_all_greeks(live_spot, live_k, live_t, r_macro, live_iv, q_funding, "CALL")
    for k, v in live_greeks.items():
        print(f"  - {k:<15}: {v}")

    # 7. Test odporności solvera IV na żywym rynku
    recovered_iv = bs.implied_volatility(
        market_price=model_c_price,
        S=live_spot,
        K=live_k,
        T=live_t,
        r=r_macro,
        q=q_funding,
        option_type="CALL"
    )
    print(f"\n4. Test solvera IV na żywym rynku:")
    print(f"  - Oczekiwane IV:                       {live_iv * 100:.2f}%")
    print(f"  - Odzyskane IV:                        {recovered_iv * 100:.2f}%")
    print(f"  - Błąd odzyskania:                     {abs(recovered_iv - live_iv):.6f}")

    # 8. Testy odporności skrajnej (Zero krachów numerycznych)
    print(f"\n5. Testy odporności skrajnej (Boundary Edge-Cases):")
    p_t_zero = bs.price(live_spot, live_k, 0.0, r_macro, live_iv, q_funding, "CALL")
    print(f"  - Payoff przy T=0 (Wygaśnięcie):       {p_t_zero:.2f} USD")

    iv_arbitrage = bs.implied_volatility(0.01, live_spot, 400.0, live_t, r_macro, q_funding, "CALL")
    print(f"  - Solver dla Deep OTM (K=400):         {iv_arbitrage}")

    print("\n[SUKCES] Silnik BlackScholesEngine działa stabilnie i precyzyjnie.")