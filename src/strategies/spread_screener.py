import os
import sys
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_feed.deribit_client import DeribitClient
from src.data_feed.market_spot import MarketSpotClient
from src.quant.volatility import VolatilityEngine
from src.quant.black_scholes import BlackScholesEngine
from src.quant.greeks import PortfolioGreeksEngine
from src.quant.payoff import PayoffEngine
from src.quant.projection import PriceProjectionEngine

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class SpreadScreener:
    """
    Skaner ilosciowy opcji SOL oparty w 100 procentach na rynku Deribit.
    W pelni eliminuje ceny widmowe z arkusza, wymusza rynek dwustronny,
    weryfikuje monotonicznosc cenowa oraz relatywny spread Bid-Ask.
    Prawidlowo rozdziela koszt carry SOL (staking yield) od sentymentu perp funding.
    Zintegrowany z interaktywnym silnikiem projekcji prawdopodobienstwa.
    """

    def __init__(
        self,
        currency: str = "SOL",
        min_dte: float = 18.0,
        max_dte: float = 40.0,
        spot_band_pct: float = 0.20,
        min_wing_width: float = 4.0,
        max_wing_width: float = 25.0,
        min_pop_pct: float = 38.0,
        max_bid_ask_spread_pct: float = 0.25
    ):
        self.currency = currency.upper()
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.exchange = "deribit"
        self.spot_band_pct = spot_band_pct
        self.min_wing_width = min_wing_width
        self.max_wing_width = max_wing_width
        self.min_pop_pct = min_pop_pct
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct

        self.deribit_client = DeribitClient()
        self.spot_client = MarketSpotClient()

    def fetch_live_market_state(self) -> Dict[str, Any]:
        spot_price = self.deribit_client.get_spot_price(self.currency)

        # Rzeczywisty koszt carry dla opcji indeksowych SOL rozliczanych do spotu (Staking Yield netto)
        q_sol_carry = float(os.getenv("SOL_STAKING_YIELD", "0.068"))
        r_risk_free = float(self.spot_client.get_us_treasury_risk_free_rate())

        # Funding rate z rynku perp pobieramy wylacznie jako indykator sentymentu
        funding_info = self.spot_client.get_current_funding_rate(f"{self.currency}USDT")
        perp_funding_apr = float(funding_info.get("annualized_rate", 0.0))

        df_candles = self.spot_client.get_clean_spot_candles(
            symbol=f"{self.currency}USDC", interval="1h", limit=168, save_csv=False
        )

        rv_7d = VolatilityEngine.garman_klass(df_candles) if not df_candles.empty else 0.60
        bv_7d = VolatilityEngine.jump_robust_bipower_variation(df_candles) if not df_candles.empty else 0.60
        rolling_rv = VolatilityEngine.rolling_garman_klass(df_candles, window=24) if not df_candles.empty else pd.Series()
        rv_24h = float(rolling_rv.iloc[-1]) if len(rolling_rv) > 0 else rv_7d

        regime_data = VolatilityEngine.classify_volatility_regime(rolling_rv, current_rv=rv_24h)

        return {
            "spot_price": spot_price,
            "r_risk_free": r_risk_free,
            "q_carry": q_sol_carry,
            "perp_funding_apr": perp_funding_apr,
            "rv_24h": rv_24h,
            "rv_7d": rv_7d,
            "bv_7d": bv_7d,
            "regime": regime_data["regime"],
            "regime_q25": regime_data["q25"],
            "regime_median": regime_data["median"],
            "regime_q75": regime_data["q75"]
        }

    def _extract_leg_price(self, leg: Dict[str, Any], is_buy: bool) -> float:
        """
        Rygorystyczna weryfikacja plynnosci arkusza bez podstawiania cen widmowych.
        Wymaga pelnego rynku dwustronnego (zarowno Bid > 0, jak i Ask > 0)
        oraz rygorystycznego spelnienia limitu relatywnego spreadu Bid-Ask.
        Kupno (BUY) wymaga realnego ask > 0.
        Sprzedaz (SELL) wymaga realnego bid > 0.
        """
        bid = leg.get("bid")
        ask = leg.get("ask")

        bid_f = float(bid) if (bid is not None and not pd.isna(bid) and float(bid) > 0.0) else 0.0
        ask_f = float(ask) if (ask is not None and not pd.isna(ask) and float(ask) > 0.0) else 0.0

        # KRYTYCZNE ZABEZPIECZENIE: Odrzucenie rynkow jednostronnych (brak bida lub brak aska)
        if bid_f <= 0.0 or ask_f <= 0.0:
            return 0.0

        # Zabezpieczenie przed skrzyzowanym arkuszem (Crossed Book / Stale Quote)
        if ask_f < bid_f:
            return 0.0

        # Enforcowany filtr szerokosci spreadu wzgledem ceny Ask
        spread_pct = (ask_f - bid_f) / ask_f
        if spread_pct > self.max_bid_ask_spread_pct:
            return 0.0

        return ask_f if is_buy else bid_f

    def generate_all_directional_spreads(
        self,
        df_deribit: pd.DataFrame,
        market_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        candidates = []
        spot = market_state.get("spot_price")
        if spot is None or spot <= 0.0:
            return []

        r = market_state["r_risk_free"]
        q = market_state["q_carry"]

        lower_strike_bound = spot * (1.0 - self.spot_band_pct)
        upper_strike_bound = spot * (1.0 + self.spot_band_pct)

        unique_expiries = df_deribit["expiry_str"].unique()

        for exp in unique_expiries:
            exp_df = df_deribit[df_deribit["expiry_str"] == exp].copy()
            if exp_df.empty:
                continue

            dte = float(exp_df["dte"].iloc[0])
            T = float(exp_df["T"].iloc[0])

            # 1. BULL CALL SPREADY
            calls = exp_df[exp_df["option_type"] == "CALL"].sort_values(by="strike").to_dict("records")
            n_calls = len(calls)

            for i in range(n_calls):
                long_leg = calls[i]
                k_long = float(long_leg["strike"])
                if not (lower_strike_bound <= k_long <= upper_strike_bound):
                    continue

                p_long = self._extract_leg_price(long_leg, is_buy=True)
                if p_long <= 0.0:
                    continue

                for j in range(i + 1, n_calls):
                    short_leg = calls[j]
                    k_short = float(short_leg["strike"])
                    if k_short > upper_strike_bound:
                        continue

                    width = k_short - k_long
                    if not (self.min_wing_width <= width <= self.max_wing_width):
                        continue

                    p_short = self._extract_leg_price(short_leg, is_buy=False)
                    if p_short <= 0.0:
                        continue

                    # KRYTYCZNE: Monotonicznosc cenowa opcji Call (C(K1) > C(K2) dla K1 < K2)
                    if p_long <= p_short:
                        continue

                    fair_iv_long = float(long_leg.get("iv") or 0.60)
                    fair_iv_short = float(short_leg.get("iv") or 0.60)

                    legs_structure = [
                        {
                            "strike": k_long,
                            "option_type": "CALL",
                            "side": "BUY",
                            "amount": 1.0,
                            "entry_price": p_long,
                            "iv": fair_iv_long,
                            "T": T
                        },
                        {
                            "strike": k_short,
                            "option_type": "CALL",
                            "side": "SELL",
                            "amount": 1.0,
                            "entry_price": p_short,
                            "iv": fair_iv_short,
                            "T": T
                        }
                    ]

                    # Wycena z prawidlowym kosztem carry aktywa bazowego
                    payoff_res = PayoffEngine.analyze_spread_payoff(
                        legs=legs_structure, S=spot, T=T, r=r, q=q, fair_iv=None, exchange=self.exchange
                    )

                    max_win = payoff_res["max_profit_usd"]
                    cap_risk = payoff_res["capital_at_risk_usd"]
                    rr_ratio = payoff_res["reward_to_risk_ratio"]
                    pop_pct = payoff_res["probability_of_profit_pct"]

                    if not isinstance(max_win, (int, float)) or max_win <= 0.0:
                        continue
                    if not isinstance(cap_risk, (int, float)) or cap_risk <= 0.0:
                        continue
                    if rr_ratio is None or rr_ratio <= 0.0:
                        continue
                    if pop_pct < self.min_pop_pct:
                        continue

                    greeks_res = PortfolioGreeksEngine.aggregate_spread_greeks(
                        legs=legs_structure, S=spot, T=T, r=r, q=q
                    )

                    candidates.append({
                        "STRATEGY": "BULL_CALL",
                        "STRAT": "BULL_CALL",
                        "EXPIRY": exp,
                        "DTE": round(dte, 1),
                        "LONG_K": k_long,
                        "SHORT_K": k_short,
                        "WIDTH": round(width, 1),
                        "NET_COST": round(payoff_res["net_entry_cost_usd"], 2),
                        "CAP_RISK": round(cap_risk, 2),
                        "MAX_WIN": round(max_win, 2),
                        "RR_RATIO": round(rr_ratio, 2),
                        "BREAKEVEN": round(payoff_res["breakeven_points"][0], 2) if payoff_res["breakeven_points"] else None,
                        "POP_PCT": round(pop_pct, 1),
                        "EV_USD": round(payoff_res.get("expected_value_usd", 0.0), 2),
                        "EXP_ROI_PCT": round(payoff_res.get("expected_roi_pct", 0.0), 1) if payoff_res.get("expected_roi_pct") is not None else None,
                        "NET_DELTA": round(greeks_res["net_delta"], 3),
                        "NET_THETA": round(greeks_res["net_theta_daily"], 3),
                        "AVG_IV": round((fair_iv_long + fair_iv_short) / 2.0, 4)
                    })

            # 2. BEAR PUT SPREADY
            puts = exp_df[exp_df["option_type"] == "PUT"].sort_values(by="strike").to_dict("records")
            n_puts = len(puts)

            for i in range(n_puts):
                # Noga krotka: nizszy strike sprzedawany
                short_leg_p = puts[i]
                k_short_p = float(short_leg_p["strike"])
                if not (lower_strike_bound <= k_short_p <= upper_strike_bound):
                    continue

                p_short_p = self._extract_leg_price(short_leg_p, is_buy=False)
                if p_short_p <= 0.0:
                    continue

                for j in range(i + 1, n_puts):
                    # Noga dluga: wyzszy strike kupowany
                    long_leg_p = puts[j]
                    k_long_p = float(long_leg_p["strike"])
                    if k_long_p > upper_strike_bound:
                        continue

                    width_p = k_long_p - k_short_p
                    if not (self.min_wing_width <= width_p <= self.max_wing_width):
                        continue

                    p_long_p = self._extract_leg_price(long_leg_p, is_buy=True)
                    if p_long_p <= 0.0:
                        continue

                    # KRYTYCZNE: Monotonicznosc cenowa opcji Put (P(K2) > P(K1) dla K2 > K1)
                    if p_long_p <= p_short_p:
                        continue

                    fair_iv_long_p = float(long_leg_p.get("iv") or 0.60)
                    fair_iv_short_p = float(short_leg_p.get("iv") or 0.60)

                    legs_structure_p = [
                        {
                            "strike": k_long_p,
                            "option_type": "PUT",
                            "side": "BUY",
                            "amount": 1.0,
                            "entry_price": p_long_p,
                            "iv": fair_iv_long_p,
                            "T": T
                        },
                        {
                            "strike": k_short_p,
                            "option_type": "PUT",
                            "side": "SELL",
                            "amount": 1.0,
                            "entry_price": p_short_p,
                            "iv": fair_iv_short_p,
                            "T": T
                        }
                    ]

                    payoff_res_p = PayoffEngine.analyze_spread_payoff(
                        legs=legs_structure_p, S=spot, T=T, r=r, q=q, fair_iv=None, exchange=self.exchange
                    )

                    max_win_p = payoff_res_p["max_profit_usd"]
                    cap_risk_p = payoff_res_p["capital_at_risk_usd"]
                    rr_ratio_p = payoff_res_p["reward_to_risk_ratio"]
                    pop_pct_p = payoff_res_p["probability_of_profit_pct"]

                    if not isinstance(max_win_p, (int, float)) or max_win_p <= 0.0:
                        continue
                    if not isinstance(cap_risk_p, (int, float)) or cap_risk_p <= 0.0:
                        continue
                    if rr_ratio_p is None or rr_ratio_p <= 0.0:
                        continue
                    if pop_pct_p < self.min_pop_pct:
                        continue

                    greeks_res_p = PortfolioGreeksEngine.aggregate_spread_greeks(
                        legs=legs_structure_p, S=spot, T=T, r=r, q=q
                    )

                    candidates.append({
                        "STRATEGY": "BEAR_PUT",
                        "STRAT": "BEAR_PUT",
                        "EXPIRY": exp,
                        "DTE": round(dte, 1),
                        "LONG_K": k_long_p,
                        "SHORT_K": k_short_p,
                        "WIDTH": round(width_p, 1),
                        "NET_COST": round(payoff_res_p["net_entry_cost_usd"], 2),
                        "CAP_RISK": round(cap_risk_p, 2),
                        "MAX_WIN": round(max_win_p, 2),
                        "RR_RATIO": round(rr_ratio_p, 2),
                        "BREAKEVEN": round(payoff_res_p["breakeven_points"][0], 2) if payoff_res_p["breakeven_points"] else None,
                        "POP_PCT": round(pop_pct, 1),
                        "EV_USD": round(payoff_res_p.get("expected_value_usd", 0.0), 2),
                        "EXP_ROI_PCT": round(payoff_res_p.get("expected_roi_pct", 0.0), 1) if payoff_res_p.get("expected_roi_pct") is not None else None,
                        "NET_DELTA": round(greeks_res_p["net_delta"], 3),
                        "NET_THETA": round(greeks_res_p["net_theta_daily"], 3),
                        "AVG_IV": round((fair_iv_long_p + fair_iv_short_p) / 2.0, 4)
                    })

        return candidates

    def run_screener(
        self,
        output_csv_filename: str = "deribit_sol_spreads.csv",
        top_n_per_strategy: int = 10
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        market_state = self.fetch_live_market_state()
        spot = market_state.get("spot_price")
        if spot is None or spot <= 0.0:
            print("[BLAD] Brak poprawnego kursu spot z Deribit.")
            return market_state, pd.DataFrame(), pd.DataFrame()

        df_deribit = self.deribit_client.get_clean_options_chain(
            underlying=self.currency, min_dte=self.min_dte, max_dte=self.max_dte
        )

        if df_deribit.empty:
            print(f"[INFO] Brak opcji na Deribit w oknie DTE ({self.min_dte} - {self.max_dte} dni).")
            return market_state, pd.DataFrame(), pd.DataFrame()

        raw_candidates = self.generate_all_directional_spreads(
            df_deribit=df_deribit, market_state=market_state
        )

        if not raw_candidates:
            print("[INFO] Brak spreadow spelniajacych kryteria na Deribit.")
            return market_state, pd.DataFrame(), pd.DataFrame()

        df_all = pd.DataFrame(raw_candidates)

        # Zapis pelnej bazy do pliku CSV
        output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, output_csv_filename)
        df_all.to_csv(csv_path, index=False)
        print(f"\n[OK] Zapisano pelna baze ({len(df_all)} spreadow z Deribit) do:")
        print(f"     -> {csv_path}\n")

        # Rozdzielenie na Top 10 Bull Call oraz Top 10 Bear Put posortowane po R/R
        df_bull_call = df_all[df_all["STRATEGY"] == "BULL_CALL"].sort_values(
            by=["RR_RATIO", "POP_PCT"], ascending=[False, False]
        ).head(top_n_per_strategy).reset_index(drop=True)

        df_bear_put = df_all[df_all["STRATEGY"] == "BEAR_PUT"].sort_values(
            by=["RR_RATIO", "POP_PCT"], ascending=[False, False]
        ).head(top_n_per_strategy).reset_index(drop=True)

        return market_state, df_bull_call, df_bear_put

    @staticmethod
    def _format_dataframe_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "STRAT": df["STRATEGY"],
            "EXPIRY": df["EXPIRY"],
            "DTE": df["DTE"].map("{:.1f}d".format),
            "LONG_K": df["LONG_K"].map("{:.0f} USD".format),
            "SHORT_K": df["SHORT_K"].map("{:.0f} USD".format),
            "WIDTH": df["WIDTH"].map("{:.0f} USD".format),
            "COST": df["NET_COST"].map("{:+.2f} USD".format),
            "RISK": df["CAP_RISK"].map("{:.2f} USD".format),
            "MAX_WIN": df["MAX_WIN"].map("{:.2f} USD".format),
            "R/R": df["RR_RATIO"].map("{:.2f}".format),
            "BREAKEVEN": df["BREAKEVEN"].apply(lambda x: f"{x:.2f} USD" if x is not None else "N/A"),
            "POP": df["POP_PCT"].map("{:.1f}%".format),
            "EV": df["EV_USD"].map("{:+.2f} USD".format),
            "EXP_ROI": df["EXP_ROI_PCT"].apply(lambda x: f"{x:+.1f}%" if (x is not None and not pd.isna(x)) else "N/A"),
            "DELTA": df["NET_DELTA"].map("{:+.3f}".format),
            "THETA": df["NET_THETA"].map("{:+.3f}".format)
        })

    @classmethod
    def display_dashboard(
        cls,
        market_state: Dict[str, Any],
        df_bull_call: pd.DataFrame,
        df_bear_put: pd.DataFrame
    ):
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        spot = market_state.get("spot_price", 0.0)

        print("========================================================================================================================")
        print(f"                         DERIBIT OFFICIAL SPREAD SCREENER (SOL) - {now_utc}")
        print("========================================================================================================================")
        print(f"Spot SOL:           {spot:.2f} USD")
        print(f"Risk-Free Rate (r): {market_state['r_risk_free'] * 100:.2f}% | SOL Carry Yield (q): {market_state['q_carry'] * 100:.2f}%")
        print(f"Perp Funding APR:   {market_state['perp_funding_apr'] * 100:.4f}% (Wskaznik sentymentu)")
        print(f"Realized Vol (RV):  24h = {market_state['rv_24h'] * 100:.2f}% | 7d GK = {market_state['rv_7d'] * 100:.2f}%")
        print("========================================================================================================================")

        # TABELA 1: BULL CALL SPREADS
        print("\n>>> DERIBIT: TOP 10 BULL CALL SPREADOW (STRATEGIE BYCZE / NA WZROST) - RANKING WG R/R <<<")
        view_call = cls._format_dataframe_view(df_bull_call)
        if view_call.empty:
            print("[INFO] Brak konfiguracji Bull Call spelniajacych kryteria na Deribit.")
        else:
            if HAS_TABULATE:
                print(tabulate(view_call, headers="keys", tablefmt="fancy_grid", showindex=True))
            else:
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", 1000)
                print(view_call.to_string(index=True))

        # TABELA 2: BEAR PUT SPREADS
        print("\n>>> DERIBIT: TOP 10 BEAR PUT SPREADOW (STRATEGIE NIEDZWIEDZIE / NA SPADEK) - RANKING WG R/R <<<")
        view_put = cls._format_dataframe_view(df_bear_put)
        if view_put.empty:
            print("[INFO] Brak konfiguracji Bear Put spelniajacych kryteria na Deribit.")
        else:
            if HAS_TABULATE:
                print(tabulate(view_put, headers="keys", tablefmt="fancy_grid", showindex=True))
            else:
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", 1000)
                print(view_put.to_string(index=True))

        print("\n========================================================================================================================")


if __name__ == "__main__":
    screener = SpreadScreener(
        currency="SOL",
        min_dte=18.0,
        max_dte=40.0,
        spot_band_pct=0.20,
        min_wing_width=4.0,
        max_wing_width=25.0,
        min_pop_pct=38.0,
        max_bid_ask_spread_pct=0.25
    )

    state, top_bull_calls, top_bear_puts = screener.run_screener(
        output_csv_filename="deribit_sol_spreads.csv",
        top_n_per_strategy=10
    )
    SpreadScreener.display_dashboard(state, top_bull_calls, top_bear_puts)

    # =========================================================================
    # INTERAKTYWNA PETLA PIPELINE'U PROJEKCJI
    # =========================================================================
    spot_val = state.get("spot_price", 120.0)
    
    # KRYTYCZNA POPRAWKA: W fizycznej projekcji trajektorii spotu dryf wynosi 0.0 (brak wypaczen z rynku perp)
    drift_val = 0.0

    while True:
        try:
            cmd = input("\nWpisz pozycje do projekcji (np. 'bull 0', 'bear 6', lub 'q' aby zakonczyc): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nZamknieto sesje.")
            break

        if cmd in ("q", "quit", "exit"):
            print("Zakonczenie pracy skanera.")
            break

        parts = cmd.split()
        if len(parts) != 2 or parts[0] not in ("bull", "bear") or not parts[1].isdigit():
            print("[BLAD] Niepoprawny format. Wpisz np. 'bull 0' albo 'bear 2'.")
            continue

        strat_type = parts[0]
        idx = int(parts[1])

        target_df = top_bull_calls if strat_type == "bull" else top_bear_puts

        if idx >= len(target_df) or idx < 0:
            print(f"[BLAD] Indeks poza zakresem. Dostepe indeksy: od 0 do {len(target_df) - 1}.")
            continue

        selected_row = target_df.iloc[idx].to_dict()
        iv_val = float(selected_row.get("AVG_IV") or 0.60)

        # Wywolanie silnika projekcji cenowej i szansy dotkniecia
        PriceProjectionEngine.display_projection_report(
            spot=spot_val,
            volatility=iv_val,
            drift=drift_val,
            strategy_row=selected_row
        )