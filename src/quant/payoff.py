import os
import sys
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.quant.black_scholes import BlackScholesEngine
from src.quant.greeks import PortfolioGreeksEngine


class PayoffEngine:
    """
    Rygorystyczny silnik analityczny profilu wyplaty, kalkulacji punktow przelamania (Breakeven),
    metryk asymetrii ryzyka do zysku (Risk/Reward), prawdopodobienstwa zysku (PoP),
    wyceny martyngalowej w mierze Q oraz rzeczywistej przewagi statystycznej (Edge) w mierze P.
    W pelni zachowuje strukture usmiechu zmiennosci (Volatility Skew) dla kazdej nogi portfela.
    """

    DAYS_PER_YEAR: float = 365.0

    # Domyslne parametry prowizji gieldowych (stan na 2026 r.)
    # Derive: ~15 bps od wartosci bazowej, z limitem max 25% wartosci premii
    DERIVE_TAKER_FEE_BPS: float = 0.0015
    DERIVE_FEE_CAP_PCT: float = 0.25

    # Deribit: ~3 bps od wartosci bazowej, z limitem max 12.5% wartosci premii
    DERIBIT_TAKER_FEE_BPS: float = 0.0003
    DERIBIT_FEE_CAP_PCT: float = 0.125

    @classmethod
    def calculate_leg_fee(
        cls,
        spot: float,
        premium: float,
        amount: float = 1.0,
        exchange: str = "derive",
        fee_bps: Optional[float] = None,
        fee_cap_pct: Optional[float] = None
    ) -> float:
        """
        Oblicza prowizje transakcyjna z uwzglednieniem limitu procentowego (fee cap).
        Gwarantuje brak aberracji: dla premium <= 0 prowizja wynosi bezwzglednie 0.0 USD.
        """
        ex = exchange.lower()
        if fee_bps is not None and fee_cap_pct is not None:
            bps = fee_bps
            cap = fee_cap_pct
        elif ex == "derive":
            bps = cls.DERIVE_TAKER_FEE_BPS
            cap = cls.DERIVE_FEE_CAP_PCT
        else:
            bps = cls.DERIBIT_TAKER_FEE_BPS
            cap = cls.DERIBIT_FEE_CAP_PCT

        raw_fee = spot * bps
        max_fee = max(0.0, premium * cap)
        unit_fee = min(raw_fee, max_fee)

        return float(unit_fee * abs(amount))

    @classmethod
    def evaluate_terminal_payoff(
        cls,
        legs: List[Dict[str, Any]],
        S_T: Union[float, np.ndarray],
        net_cost: float = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Oblicza czysty PnL terminalny w momencie wygasniecia (T = 0) dla podanego kursu lub wektora kursow S_T.
        PnL = Wyplata_brutto - Koszt_netto
        """
        is_array = isinstance(S_T, np.ndarray)
        payoff = np.zeros_like(S_T, dtype=float) if is_array else 0.0

        for leg in legs:
            K = float(leg.get("strike", 0.0))
            opt_type = str(leg.get("option_type", "CALL")).upper()
            side = leg.get("side", "BUY")
            amount = float(leg.get("amount", 1.0))
            weight = PortfolioGreeksEngine.parse_leg_weight(side, amount)

            if opt_type == "CALL":
                if is_array:
                    leg_payout = np.maximum(0.0, S_T - K)
                else:
                    leg_payout = max(0.0, S_T - K)
            elif opt_type == "PUT":
                if is_array:
                    leg_payout = np.maximum(0.0, K - S_T)
                else:
                    leg_payout = max(0.0, K - S_T)
            else:
                continue

            payoff = payoff + (weight * leg_payout)

        return payoff - net_cost

    @classmethod
    def _find_exact_breakevens(
        cls,
        legs: List[Dict[str, Any]],
        net_cost: float
    ) -> List[float]:
        """
        Analityczne wyznaczenie wszystkich punktow przelamania (miejsc zerowych PnL).
        Dziala bez heurystyk numerycznych na odcinkach wewnetrznych i prawej polprostej.
        """
        strikes = sorted(list({float(leg["strike"]) for leg in legs if "strike" in leg}))
        if not strikes:
            return []

        breakevens = []

        # 1. Punkty wezlowe dla odcinkow wewnetrznych
        nodes = [0.0] + strikes
        for i in range(len(nodes) - 1):
            s_left = nodes[i]
            s_right = nodes[i + 1]

            pnl_left = float(cls.evaluate_terminal_payoff(legs, s_left, net_cost))
            pnl_right = float(cls.evaluate_terminal_payoff(legs, s_right, net_cost))

            if abs(pnl_left) < 1e-9:
                breakevens.append(round(s_left, 4))
            elif (pnl_left > 0.0 and pnl_right < 0.0) or (pnl_left < 0.0 and pnl_right > 0.0):
                slope = (pnl_right - pnl_left) / (s_right - s_left)
                if abs(slope) > 1e-12:
                    root = s_left - (pnl_left / slope)
                    if s_left <= root <= s_right:
                        breakevens.append(round(root, 4))

        # Wezel koncowy K_max
        max_k = strikes[-1]
        pnl_max_k = float(cls.evaluate_terminal_payoff(legs, max_k, net_cost))
        if abs(pnl_max_k) < 1e-9:
            breakevens.append(round(max_k, 4))

        # 2. Prawa polprosta [K_max, infinity): analityczne nachylenie
        slope_inf = 0.0
        for leg in legs:
            opt_type = str(leg.get("option_type", "CALL")).upper()
            if opt_type == "CALL":
                w = PortfolioGreeksEngine.parse_leg_weight(leg.get("side", "BUY"), leg.get("amount", 1.0))
                slope_inf += w * 1.0

        if abs(slope_inf) > 1e-12:
            root_inf = max_k - (pnl_max_k / slope_inf)
            if root_inf > max_k + 1e-7:
                breakevens.append(round(float(root_inf), 4))

        # Oczyszczenie i dedublikacja
        unique_breakevens = []
        for b in sorted(breakevens):
            if not unique_breakevens or not math.isclose(unique_breakevens[-1], b, abs_tol=1e-4):
                unique_breakevens.append(b)

        return unique_breakevens

    @classmethod
    def calculate_pop(
        cls,
        legs: List[Dict[str, Any]],
        breakevens: List[float],
        net_cost: float,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """
        Scisle wyliczenie Prawdopodobienstwa Zysku (Probability of Profit - PoP).
        Calkuje analitycznie rozklad log-normalny forwardu P(S_T in Zysk).
        """
        if T <= 1e-7 or sigma <= 1e-4 or S <= 0.0:
            pnl_now = cls.evaluate_terminal_payoff(legs, S, net_cost)
            return 1.0 if pnl_now > 0.0 else 0.0

        forward = S * math.exp((r - q) * T)
        std_dev = sigma * math.sqrt(T)

        def cdf_at(x: float) -> float:
            if x <= 0.0:
                return 0.0
            d2 = (math.log(forward / x) - 0.5 * (sigma ** 2.0) * T) / std_dev
            return float(norm.cdf(-d2))

        if not breakevens:
            sample_pnl = cls.evaluate_terminal_payoff(legs, S, net_cost)
            return 1.0 if sample_pnl > 0.0 else 0.0

        intervals = [0.0] + breakevens + [float("inf")]
        pop_total = 0.0

        for i in range(len(intervals) - 1):
            lower = intervals[i]
            upper = intervals[i + 1]

            if math.isinf(upper):
                test_point = lower + 10.0
            else:
                test_point = (lower + upper) / 2.0

            pnl_test = cls.evaluate_terminal_payoff(legs, test_point, net_cost)
            if pnl_test > 1e-7:
                p_upper = 1.0 if math.isinf(upper) else cdf_at(upper)
                p_lower = cdf_at(lower)
                pop_total += max(0.0, p_upper - p_lower)

        return round(float(min(1.0, max(0.0, pop_total))), 4)

    @classmethod
    def calculate_expected_value(
        cls,
        legs: List[Dict[str, Any]],
        S: float,
        T: float,
        r: float,
        q: float,
        fair_iv: Optional[float] = None,
        forecast_rv: Optional[float] = None,
        net_cost: float = 0.0,
        capital_at_risk: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Prawidlowe rozdzielenie wyceny martyngalowej w mierze Q od statystycznego Edge w mierze P.
        Jesli podano forecast_rv, wycenia kontrakt wedlug prognozy fizycznej zmiennosci,
        porownujac ja z rynkowym kosztem wejscia (statistical_edge_usd).
        """
        if T <= 1e-7:
            pnl_terminal = cls.evaluate_terminal_payoff(legs, S, net_cost)
            car = capital_at_risk if (capital_at_risk is not None and capital_at_risk > 0.0) else max(abs(net_cost), 1.0)
            roi = round(float(pnl_terminal / car * 100.0), 2) if car > 0.0 else 0.0
            return {
                "fair_portfolio_pv_usd": round(float(pnl_terminal + net_cost), 4),
                "expected_gross_payout_usd": round(float(pnl_terminal + net_cost), 4),
                "expected_net_ev_usd": round(float(pnl_terminal), 4),
                "statistical_edge_usd": 0.0,
                "capital_at_risk_usd": round(float(car), 4),
                "expected_roi_pct": roi
            }

        compounding_factor = math.exp(r * T)
        physical_fair_pv = 0.0

        for leg in legs:
            K = float(leg.get("strike", S))
            opt_type = str(leg.get("option_type", "CALL")).upper()
            side = leg.get("side", "BUY")
            amount = float(leg.get("amount", 1.0))
            weight = PortfolioGreeksEngine.parse_leg_weight(side, amount)

            # Wybor zmiennosci: priorytet dla prognozy fizycznej RV (miara P), nastepnie dedykowany IV nogi
            if forecast_rv is not None and forecast_rv > 0.0:
                eval_vol = forecast_rv
            else:
                eval_vol = float(leg.get("iv") or fair_iv or 0.60)

            model_price = BlackScholesEngine.price(
                S=S, K=K, T=T, r=r, sigma=eval_vol, q=q, option_type=opt_type
            )
            physical_fair_pv += weight * model_price

        # Wyplata brutto na wygasnieciu akumulowana stopa wolna od ryzyka r
        expected_gross_payout = physical_fair_pv * compounding_factor
        expected_net_ev = expected_gross_payout - net_cost
        statistical_edge = physical_fair_pv - net_cost

        # Wyznaczenie bazy kapitalu narazonego na ryzyko
        if capital_at_risk is not None and capital_at_risk > 0.0:
            car = capital_at_risk
        elif net_cost > 0.0:
            car = net_cost
        else:
            car = abs(net_cost) if abs(net_cost) > 0.0 else 1.0

        expected_roi = round(float(expected_net_ev / car * 100.0), 2) if car > 0.0 else 0.0

        return {
            "fair_portfolio_pv_usd": round(physical_fair_pv, 4),
            "expected_gross_payout_usd": round(expected_gross_payout, 4),
            "expected_net_ev_usd": round(expected_net_ev, 4),
            "statistical_edge_usd": round(statistical_edge, 4),
            "capital_at_risk_usd": round(car, 4),
            "expected_roi_pct": expected_roi
        }

    @classmethod
    def analyze_spread_payoff(
        cls,
        legs: List[Dict[str, Any]],
        S: float,
        T: float,
        r: float = 0.05,
        q: float = 0.0,
        fair_iv: Optional[float] = None,
        forecast_rv: Optional[float] = None,
        exchange: str = "derive",
        custom_fee_per_leg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Glowna metoda analityczna. Dokonuje calosciowej oceny oplacalnosci:
        Max Profit, Max Loss, Capital at Risk, R/R, Breakevens, PoP, EV, Statistical Edge oraz prowizje.
        """
        if not legs or S <= 0.0:
            return {"error": "Bledne parametry wejsciowe lub brak nog spreadu."}

        # 1. Rozliczenie przeplywow pienieznych (Cash Flow) i prowizji
        total_premium_paid = 0.0
        total_fees = 0.0
        strikes = []

        for leg in legs:
            K = float(leg.get("strike", S))
            strikes.append(K)
            amount = float(leg.get("amount", 1.0))
            side = leg.get("side", "BUY")
            weight = PortfolioGreeksEngine.parse_leg_weight(side, amount)

            leg_price = float(leg.get("entry_price") or leg.get("mark") or leg.get("price") or 0.0)
            total_premium_paid += weight * leg_price

            if custom_fee_per_leg is not None:
                total_fees += custom_fee_per_leg * abs(amount)
            else:
                total_fees += cls.calculate_leg_fee(
                    spot=S, premium=leg_price, amount=amount, exchange=exchange
                )

        net_entry_cost = total_premium_paid + total_fees
        is_debit = net_entry_cost > 0.0
        structure_type = "DEBIT_SPREAD" if is_debit else "CREDIT_SPREAD"

        # 2. Analiza asymptotyczna (Wykrywanie nieograniczonego ryzyka lub zysku)
        slope_at_inf = 0.0
        for leg in legs:
            amount = float(leg.get("amount", 1.0))
            side = leg.get("side", "BUY")
            w = PortfolioGreeksEngine.parse_leg_weight(side, amount)
            opt_type = str(leg.get("option_type", "CALL")).upper()
            if opt_type == "CALL":
                slope_at_inf += w * 1.0

        is_profit_unbounded = slope_at_inf > 1e-7
        is_loss_unbounded = slope_at_inf < -1e-7

        # 3. Badanie ekstremow w punktach osobliwych
        crit_points = [0.0] + sorted(strikes) + [max(strikes) * 2.0]
        crit_pnls = [cls.evaluate_terminal_payoff(legs, pt, net_entry_cost) for pt in crit_points]

        max_profit = float("inf") if is_profit_unbounded else max(crit_pnls)
        max_loss = -float("inf") if is_loss_unbounded else min(crit_pnls)

        # 4. Wyznaczenie kapitalu narazonego na ryzyko (Capital at Risk)
        if is_loss_unbounded:
            capital_at_risk = float("inf")
        elif is_debit:
            capital_at_risk = net_entry_cost
        else:
            capital_at_risk = abs(max_loss) if abs(max_loss) > 0.0 else abs(net_entry_cost)

        # 5. Wyznaczenie wskaznikow asymetrii Risk / Reward
        if math.isinf(max_profit) or math.isinf(max_loss) or max_profit <= 0.0 or max_loss >= 0.0:
            risk_reward_ratio = None
            reward_to_risk_ratio = None
        else:
            risk = abs(max_loss)
            reward = max_profit
            risk_reward_ratio = round(risk / reward, 4) if reward > 0.0 else None
            reward_to_risk_ratio = round(reward / risk, 4) if risk > 0.0 else None

        # 6. Analityczne Breakevens
        breakevens = cls._find_exact_breakevens(legs, net_entry_cost)

        # 7. Prawdopodobieństwo Zysku (PoP) z uśrednionego IV obu nóg spreadu
        leg_ivs = [float(l.get("iv")) for l in legs if l.get("iv") is not None and float(l.get("iv")) > 0.0]
        if fair_iv is not None and fair_iv > 0.0:
            ref_iv = fair_iv
        elif leg_ivs:
            ref_iv = sum(leg_ivs) / len(leg_ivs)
        else:
            ref_iv = 0.60
        pop = cls.calculate_pop(
            legs=legs,
            breakevens=breakevens,
            net_cost=net_entry_cost,
            S=S,
            T=T,
            r=r,
            q=q,
            sigma=ref_iv
        )

        # 8. Wartosc Oczekiwana (EV) oraz Przewaga Statystyczna (Edge)
        ev_metrics = cls.calculate_expected_value(
            legs=legs,
            S=S,
            T=T,
            r=r,
            q=q,
            fair_iv=fair_iv,
            forecast_rv=forecast_rv,
            net_cost=net_entry_cost,
            capital_at_risk=capital_at_risk if not math.isinf(capital_at_risk) else None
        )

        return {
            "structure_type": structure_type,
            "net_premium_usd": round(total_premium_paid, 4),
            "total_fees_usd": round(total_fees, 4),
            "net_entry_cost_usd": round(net_entry_cost, 4),
            "capital_at_risk_usd": round(capital_at_risk, 4) if not math.isinf(capital_at_risk) else "UNBOUNDED",
            "max_profit_usd": round(max_profit, 4) if not math.isinf(max_profit) else "UNBOUNDED",
            "max_loss_usd": round(max_loss, 4) if not math.isinf(max_loss) else "UNBOUNDED",
            "is_profit_unbounded": is_profit_unbounded,
            "is_loss_unbounded": is_loss_unbounded,
            "risk_reward_ratio": risk_reward_ratio,
            "reward_to_risk_ratio": reward_to_risk_ratio,
            "breakeven_points": breakevens,
            "probability_of_profit_pct": round(pop * 100.0, 2),
            "expected_value_usd": ev_metrics["expected_net_ev_usd"],
            "statistical_edge_usd": ev_metrics["statistical_edge_usd"],
            "expected_roi_pct": ev_metrics["expected_roi_pct"] if not math.isinf(capital_at_risk) else None,
            "is_ev_positive": ev_metrics["expected_net_ev_usd"] > 0.0,
            "is_edge_positive": ev_metrics["statistical_edge_usd"] > 0.0
        }

    @classmethod
    def generate_payoff_curve(
        cls,
        legs: List[Dict[str, Any]],
        S: float,
        T: float,
        sigma: float = 0.60,
        net_cost: Optional[float] = None,
        num_points: int = 250
    ) -> pd.DataFrame:
        """
        Generuje nieliniowa, gesta wektorowa siatke profilu PnL w calym spektrum cenowym.
        """
        if net_cost is None:
            net_cost = sum(
                PortfolioGreeksEngine.parse_leg_weight(l.get("side", "BUY"), l.get("amount", 1.0)) *
                float(l.get("entry_price") or l.get("mark") or 0.0)
                for l in legs
            )

        strikes = [float(l["strike"]) for l in legs if "strike" in l]
        min_k = min(strikes) if strikes else S * 0.8
        max_k = max(strikes) if strikes else S * 1.2

        std_range = sigma * math.sqrt(max(T, 7.0 / cls.DAYS_PER_YEAR))
        s_min = max(1.0, min(S * math.exp(-3.5 * std_range), min_k * 0.75))
        s_max = max(S * math.exp(3.5 * std_range), max_k * 1.25)

        grid_base = np.linspace(s_min, s_max, num_points)
        special_pts = np.array([S] + strikes)
        full_grid = np.unique(np.sort(np.concatenate([grid_base, special_pts])))

        pnl_values = cls.evaluate_terminal_payoff(legs, full_grid, net_cost)

        df = pd.DataFrame({
            "spot_price": np.round(full_grid, 2),
            "net_pnl": np.round(pnl_values, 4),
            "roi_pct": np.round((pnl_values / net_cost * 100.0) if net_cost > 0 else 0.0, 2)
        })
        return df