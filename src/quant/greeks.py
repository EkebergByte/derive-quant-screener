import os
import sys
import math
from typing import Dict, Any, List, Optional, Union
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.quant.black_scholes import BlackScholesEngine


class PortfolioGreeksEngine:
    """
    Zaawansowany agregator grek portfelowych i wielonogowych struktur opcyjnych (Spready,
    Kalendarze, Diagonale, Kondory, Strangle). Zapewnia pełną spójność wymiarową jednostek,
    obsługuje niezależne terminy wygaśnięcia dla poszczególnych nóg oraz realizuje
    rygorystyczną dekompozycję PnL drugiego rzędu (Taylor: Delta-Gamma-Theta-Vega-Vanna-Volga).
    """

    DAYS_PER_YEAR: float = 365.0  # Standard krypto ACT/365 (spójny z Deribit i Derive)

    @classmethod
    def parse_leg_weight(cls, side: Union[str, int, float], amount: float = 1.0) -> float:
        """
        Zwraca znakowaną wagę pozycji w strukturze:
        +amount dla pozycji długiej (BUY / LONG),
        -amount dla pozycji krótkiej (SELL / SHORT).
        """
        if isinstance(side, (int, float)):
            sign = 1.0 if side >= 0 else -1.0
        elif isinstance(side, str):
            sign = 1.0 if side.upper() in ("BUY", "LONG", "+", "+1") else -1.0
        else:
            sign = 1.0
        return float(sign * abs(amount))

    @classmethod
    def resolve_leg_t(cls, leg: Dict[str, Any], global_t: Optional[float] = None) -> float:
        """
        Pomocnicza funkcja rozwiązująca czas do wygaśnięcia T dla pojedynczej nogi.
        Priorytet: leg['T'] -> leg['dte'] -> global_t.
        """
        if "T" in leg and leg["T"] is not None:
            return max(0.0, float(leg["T"]))
        if "dte" in leg and leg["dte"] is not None:
            return max(0.0, float(leg["dte"]) / cls.DAYS_PER_YEAR)
        if global_t is not None:
            return max(0.0, float(global_t))
        raise ValueError(f"Nie podano horyzontu czasowego (T lub dte) dla nogi: {leg}")

    @classmethod
    def aggregate_spread_greeks(
        cls,
        legs: List[Dict[str, Any]],
        S: float,
        T: Optional[float] = None,
        r: float = 0.05,
        q: float = 0.0
    ) -> Dict[str, Any]:
        """
        Agreguje greki portfela opcyjnego. Obsługuje zarówno struktury o wspólnym terminie
        wygaśnięcia (vertical spreads), jak i różne terminy wygaśnięcia poszczególnych nóg
        (calendar / diagonal spreads).
        """
        if not legs or S <= 0.0:
            return {
                "net_price": 0.0,
                "net_delta": 0.0,
                "net_gamma": 0.0,
                "net_theta_daily": 0.0,
                "net_vega_1pct": 0.0,
                "net_rho_1pct": 0.0,
                "net_vanna_1pct": 0.0,
                "net_volga_1pct": 0.0,
                "legs_breakdown": []
            }

        total_net_price = 0.0
        total_net_delta = 0.0
        total_net_gamma = 0.0
        total_net_theta = 0.0
        total_net_vega = 0.0
        total_net_rho = 0.0
        total_net_vanna = 0.0
        total_net_volga = 0.0

        legs_detail = []

        for leg in legs:
            K = float(leg.get("strike", S))
            opt_type = str(leg.get("option_type", "CALL")).upper()
            iv = float(leg.get("iv", 0.60))
            amount = float(leg.get("amount", 1.0))
            side = leg.get("side", "BUY")

            leg_t = cls.resolve_leg_t(leg, T)
            weight = cls.parse_leg_weight(side, amount)

            # Analityczne greki pojedynczego kontraktu z silnika BlackScholesEngine
            g = BlackScholesEngine.get_all_greeks(S=S, K=K, T=leg_t, r=r, sigma=iv, q=q, option_type=opt_type)

            leg_price_weighted = g["price"] * weight
            leg_delta_weighted = g["delta"] * weight
            leg_gamma_weighted = g["gamma"] * weight
            leg_theta_weighted = g["theta_daily"] * weight
            leg_vega_weighted = g["vega_1pct"] * weight
            leg_rho_weighted = g["rho_1pct"] * weight
            leg_vanna_weighted = g["vanna_1pct"] * weight
            leg_volga_weighted = g["volga_1pct"] * weight

            total_net_price += leg_price_weighted
            total_net_delta += leg_delta_weighted
            total_net_gamma += leg_gamma_weighted
            total_net_theta += leg_theta_weighted
            total_net_vega += leg_vega_weighted
            total_net_rho += leg_rho_weighted
            total_net_vanna += leg_vanna_weighted
            total_net_volga += leg_volga_weighted

            legs_detail.append({
                "strike": K,
                "option_type": opt_type,
                "side": "BUY" if weight > 0 else "SELL",
                "amount": abs(amount),
                "iv": iv,
                "T": leg_t,
                "dte": round(leg_t * cls.DAYS_PER_YEAR, 2),
                "single_price": g["price"],
                "weighted_price": round(leg_price_weighted, 4),
                "single_delta": g["delta"],
                "weighted_delta": round(leg_delta_weighted, 4),
                "single_gamma": g["gamma"],
                "weighted_gamma": round(leg_gamma_weighted, 6),
                "single_theta": g["theta_daily"],
                "weighted_theta": round(leg_theta_weighted, 4),
                "single_vega": g["vega_1pct"],
                "weighted_vega": round(leg_vega_weighted, 4),
                "single_vanna": g["vanna_1pct"],
                "weighted_vanna": round(leg_vanna_weighted, 6),
                "single_volga": g["volga_1pct"],
                "weighted_volga": round(leg_volga_weighted, 6)
            })

        return {
            "net_price": round(total_net_price, 4),
            "net_delta": round(total_net_delta, 4),
            "net_gamma": round(total_net_gamma, 6),
            "net_theta_daily": round(total_net_theta, 4),
            "net_vega_1pct": round(total_net_vega, 4),
            "net_rho_1pct": round(total_net_rho, 4),
            "net_vanna_1pct": round(total_net_vanna, 6),
            "net_volga_1pct": round(total_net_volga, 6),
            "legs_breakdown": legs_detail
        }

    @classmethod
    def scenario_stress_test(
        cls,
        legs: List[Dict[str, Any]],
        S: float,
        T: Optional[float] = None,
        r: float = 0.05,
        q: float = 0.0,
        price_shift_pct: float = 0.0,
        iv_shift_pts: float = 0.0,
        days_forward: float = 0.0
    ) -> Dict[str, Any]:
        """
        Nieliniowa symulacja warunków skrajnych (What-If Analysis).
        Przelicza portfel przy wstrząsie cenowym spot, szoku zmienności IV oraz upływie czasu.
        Zapewnia pełną spójność wymiarową aproksymacji Taylora drugiego rzędu
        (Delta, Gamma, Vega, Theta, Vanna, Volga) z rzeczywistym repricingiem.
        """
        base_state = cls.aggregate_spread_greeks(legs=legs, S=S, T=T, r=r, q=q)
        base_value = base_state["net_price"]

        # 1. Wektor szoków rynkowych
        new_spot = S * (1.0 + price_shift_pct)
        delta_s = new_spot - S
        delta_sigma = iv_shift_pts  # Zmiana ułamkowa IV (np. +0.05 to +5 p.p. IV)

        # 2. Pełny repricing numeryczny (Full Non-Linear Repricing) z niezależnym czasem dla każdej nogi
        new_legs = []
        theta_taylor_pnl = 0.0

        for leg_info in base_state["legs_breakdown"]:
            leg_t_base = leg_info["T"]
            leg_max_days = leg_t_base * cls.DAYS_PER_YEAR

            # Bezwzględne zabezpieczenie przed erozją czasową po wygaśnięciu
            effective_leg_days = min(days_forward, leg_max_days)
            time_decrement = effective_leg_days / cls.DAYS_PER_YEAR
            new_leg_t = max(0.0, leg_t_base - time_decrement)

            # Erozja czasowa w Taylorze per noga (ograniczona do terminu wygaśnięcia)
            theta_taylor_pnl += leg_info["weighted_theta"] * effective_leg_days

            new_leg = {
                "strike": leg_info["strike"],
                "option_type": leg_info["option_type"],
                "side": leg_info["side"],
                "amount": leg_info["amount"],
                "T": new_leg_t,
                "iv": max(BlackScholesEngine.MIN_SIGMA, min(BlackScholesEngine.MAX_SIGMA, leg_info["iv"] + iv_shift_pts))
            }
            new_legs.append(new_leg)

        shocked_state = cls.aggregate_spread_greeks(legs=new_legs, S=new_spot, T=None, r=r, q=q)
        new_value = shocked_state["net_price"]
        exact_pnl = new_value - base_value

        # 3. Spójna wymiarowo aproksymacja Taylora drugiego rzędu
        # Przywracamy surowe pochodne cząstkowe z wartości znormalizowanych do 1 p.p.
        raw_vega = base_state["net_vega_1pct"] * 100.0       # dV / dSigma
        raw_vanna = base_state["net_vanna_1pct"] * 100.0     # d^2V / (dS dSigma)
        raw_volga = base_state["net_volga_1pct"] * 10000.0   # d^2V / dSigma^2

        delta_pnl = base_state["net_delta"] * delta_s
        gamma_pnl = 0.5 * base_state["net_gamma"] * (delta_s ** 2.0)
        theta_pnl = theta_taylor_pnl
        vega_pnl = raw_vega * delta_sigma
        vanna_pnl = raw_vanna * delta_s * delta_sigma
        volga_pnl = 0.5 * raw_volga * (delta_sigma ** 2.0)

        taylor_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + vanna_pnl + volga_pnl
        taylor_error = exact_pnl - taylor_pnl

        # Średnie resztkowe DTE całego portfela
        remaining_dtes = [l["dte"] for l in shocked_state["legs_breakdown"]]
        avg_remaining_dte = (sum(remaining_dtes) / len(remaining_dtes)) if remaining_dtes else 0.0

        return {
            "base_spot": round(S, 2),
            "shocked_spot": round(new_spot, 2),
            "spot_change_usd": round(delta_s, 2),
            "spot_change_pct": round(price_shift_pct * 100.0, 2),
            "iv_shift_pts": round(iv_shift_pts * 100.0, 2),
            "days_elapsed": round(days_forward, 2),
            "avg_remaining_dte": round(avg_remaining_dte, 2),
            "base_portfolio_value": round(base_value, 4),
            "new_portfolio_value": round(new_value, 4),
            "exact_pnl_usd": round(exact_pnl, 4),
            "taylor_pnl_approx": round(taylor_pnl, 4),
            "decomposition": {
                "delta_pnl": round(delta_pnl, 4),
                "gamma_pnl": round(gamma_pnl, 4),
                "theta_pnl": round(theta_pnl, 4),
                "vega_pnl": round(vega_pnl, 4),
                "vanna_pnl": round(vanna_pnl, 4),
                "volga_pnl": round(volga_pnl, 4)
            },
            "non_linear_residual": round(taylor_error, 4),
            "new_net_delta": shocked_state["net_delta"],
            "new_net_gamma": shocked_state["net_gamma"],
            "new_net_theta_daily": shocked_state["net_theta_daily"],
            "new_net_vega_1pct": shocked_state["net_vega_1pct"]
        }

    @classmethod
    def calculate_delta_hedge(
        cls,
        net_delta: float,
        min_hedge_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Zwraca parametry zlecenia zabezpieczającego ekspozycję kierunkową portfela.
        Uwzględnia pasmo martwe (deadband), eliminując mikrozlecenia poniżej minimalnego kroku giełdy.
        """
        if abs(net_delta) < min_hedge_threshold:
            return {
                "net_delta_to_neutralize": round(net_delta, 4),
                "hedge_action": "BALANCED",
                "hedge_amount": 0.0
            }

        hedge_units = -net_delta
        action = "BUY_SPOT_PERP" if hedge_units > 0 else "SELL_SPOT_PERP"

        return {
            "net_delta_to_neutralize": round(net_delta, 4),
            "hedge_action": action,
            "hedge_amount": round(abs(hedge_units), 4)
        }


if __name__ == "__main__":
    from src.data_feed.market_spot import MarketSpotClient
    from src.data_feed.derive_client import DeriveClient

    print("==================================================")
    print("TEST ZMODERNIZOWANEGO PORTFOLIOGREEKSENGINE W RAM")
    print("==================================================")

    spot_client = MarketSpotClient()
    derive_client = DeriveClient()

    live_spot = derive_client.get_spot_price("SOL") or 150.0
    funding_info = spot_client.get_current_funding_rate("SOLUSDT")
    q_funding = funding_info.get("annualized_rate", 0.0)
    r_macro = spot_client.get_us_treasury_risk_free_rate()

    print(f"Kurs Spot: {live_spot} USD | Stopa r: {r_macro * 100:.2f}% | Funding q: {q_funding * 100:.4f}%\n")

    # Przykładowy Bull Call Spread (Vertical)
    spread = [
        {"strike": live_spot, "option_type": "CALL", "side": "BUY", "amount": 10.0, "iv": 0.60, "dte": 14.0},
        {"strike": live_spot * 1.10, "option_type": "CALL", "side": "SELL", "amount": 10.0, "iv": 0.65, "dte": 14.0}
    ]

    base_metrics = PortfolioGreeksEngine.aggregate_spread_greeks(spread, S=live_spot, r=r_macro, q=q_funding)
    print(f"Wycena początkowa: {base_metrics['net_price']:.2f} USD")
    print(f"Delta netto:       {base_metrics['net_delta']:.4f} SOL")
    print(f"Gamma netto:       {base_metrics['net_gamma']:.6f}")
    print(f"Theta dobowa:      {base_metrics['net_theta_daily']:.4f} USD/dzień")
    print(f"Vega netto 1%:     {base_metrics['net_vega_1pct']:.4f} USD\n")

    stress_res = PortfolioGreeksEngine.scenario_stress_test(
        legs=spread,
        S=live_spot,
        r=r_macro,
        q=q_funding,
        price_shift_pct=+0.05,
        iv_shift_pts=-0.03,
        days_forward=3.0
    )

    print("--- WYNIK STRESS TESTU (+5% Spot, -3% IV, +3 dni) ---")
    print(f"Rzeczywisty PnL (Repricing): {stress_res['exact_pnl_usd']:+.4f} USD")
    print(f"Aproksymacja Taylora:        {stress_res['taylor_pnl_approx']:+.4f} USD")
    print(f"Residuum nieliniowe:         {stress_res['non_linear_residual']:+.4f} USD")
    print("Dekompozycja składowych PnL:", stress_res["decomposition"])

    hedge_info = PortfolioGreeksEngine.calculate_delta_hedge(base_metrics["net_delta"], min_hedge_threshold=0.05)
    print(f"\nDelta Hedge: {hedge_info['hedge_action']} {hedge_info['hedge_amount']} SOL")