#!/usr/bin/env python3
"""
EVE "Money Button" — Rankings Generator (Arbitrage + Emperor features)

This script generates `docs/data/rankings.json` for the static GitHub Pages site.

Key features
- Multi-market arbitrage: buy inputs in ONE "buy market", evaluate selling outputs across MANY "sell markets",
  and select the best destination (per mode: Instant vs Patient).
- Conservative execution modes:
  * Instant: buy inputs from sell orders, sell outputs into buy orders (fastest, lowest thinking).
  * Patient: buy inputs via buy orders, sell outputs via sell orders (higher margin, slower to realize).
- Confidence score (0–100) to down-rank manipulated / dead / illiquid items.
- Depth simulation for top-N rows (real orderbooks) to compute:
  * Max runs by input depth (input slippage tolerance)
  * Max runs by output depth (output slippage tolerance)
  * Recommended runs (with safety buffer)
  * "Guaranteed executable profit" label when recommended runs are still profitable after fees/taxes/slippage
- Time-to-liquidate estimate (bucketed) from ESI regional market history (where available).
- Hauling sanity metrics (m³/run, profit per m³).
- Optional EVE SSO (refresh-token) support to:
  * Read your character skills (for job time modifiers)
  * Read structure markets (null/low hub structures) via authenticated ESI

Notes / reality checks
- Nothing can *guarantee* profit in a player market; the goal is to minimize "looks good but isn't executable".
- Station/region markets are sourced from Fuzzwork aggregates for speed.
- Structure markets require SSO and can be heavy to fetch (ESI has no type filter for structure orders).

"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

# -----------------------------
# Constants / paths
# -----------------------------

ESI_BASE = "https://esi.evetech.net/latest"
ESI_DATASOURCE = "tranquility"

FUZZWORK_STATION_AGG_URL = "https://market.fuzzwork.co.uk/aggregates/"
FUZZWORK_AGGREGATECSV_URL = "https://www.fuzzwork.co.uk/dump/latest/aggregatecsv.csv.bz2"

RECIPES_PATH = Path("data/recipes.json.gz")
OUT_PATH_DEFAULT = Path("docs/data/rankings.json")
CACHE_DIR = Path(".cache")

STATION_AGG_CHUNK = 200  # max types per fuzzwork station aggregates request


# -----------------------------
# Small utilities
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def chunked(seq: List[int], n: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# -----------------------------
# Markets config
# -----------------------------

@dataclass(frozen=True)
class Market:
    key: str
    name: str
    kind: str  # "station" | "region" | "structure"
    region_id: Optional[int] = None
    station_id: Optional[int] = None
    structure_id: Optional[int] = None
    system_name: Optional[str] = None
    system_id: Optional[int] = None

    @property
    def location_id(self) -> Optional[int]:
        return self.station_id or self.structure_id

    def is_valid(self) -> bool:
        if self.kind == "station":
            return self.station_id is not None and self.region_id is not None
        if self.kind == "region":
            return self.region_id is not None
        if self.kind == "structure":
            return self.structure_id is not None and self.structure_id != 0
        return False


@dataclass
class MarketsConfig:
    markets: Dict[str, Market]
    default_buy: str
    default_sells: List[str]


def load_markets_config(path: Path) -> MarketsConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mkts: Dict[str, Market] = {}
    for key, m in (raw.get("markets") or {}).items():
        mkts[key] = Market(
            key=key,
            name=str(m.get("name") or key),
            kind=str(m.get("kind") or "station"),
            region_id=m.get("region_id"),
            station_id=m.get("station_id"),
            structure_id=m.get("structure_id"),
            system_name=m.get("system_name"),
            system_id=m.get("system_id"),
        )
    default_buy = str(raw.get("default_buy") or "jita")
    default_sells = list(raw.get("default_sells") or [])
    return MarketsConfig(markets=mkts, default_buy=default_buy, default_sells=default_sells)


def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# -----------------------------
# EVE SSO (optional)
# -----------------------------

def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def decode_jwt_no_verify(token: str) -> Dict[str, Any]:
    # Token is a JWT: header.payload.signature
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def sso_access_token_from_refresh(
    client_id: str, client_secret: str, refresh_token: str, timeout_s: int = 20
) -> Optional[str]:
    # OAuth refresh flow (confidential clients)
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "eve-money-button/1.0",
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    url = "https://login.eveonline.com/v2/oauth/token"
    try:
        r = requests.post(url, headers=headers, data=data, timeout=timeout_s)
        r.raise_for_status()
        j = r.json()
        return j.get("access_token")
    except Exception:
        return None


def fetch_character_id_and_name(access_token: str, timeout_s: int = 20) -> Tuple[Optional[int], Optional[str]]:
    # Best effort:
    # 1) decode JWT (no verify) for sub/name
    # 2) fallback to legacy oauth/verify if available
    claims = decode_jwt_no_verify(access_token)
    sub = claims.get("sub") or ""
    name = claims.get("name")
    char_id: Optional[int] = None
    if isinstance(sub, str) and "CHARACTER" in sub:
        # Often "CHARACTER:EVE:<id>"
        try:
            char_id = int(sub.split(":")[-1])
        except Exception:
            char_id = None

    if char_id and name:
        return char_id, str(name)

    # Fallback: oauth/verify (deprecated but still widely used)
    try:
        r = requests.get(
            "https://login.eveonline.com/oauth/verify",
            headers={"Authorization": f"Bearer {access_token}", "User-Agent": "eve-money-button/1.0"},
            timeout=timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        return safe_int(j.get("CharacterID"), None), j.get("CharacterName")
    except Exception:
        return char_id, str(name) if name else (None, None)


def esi_get(url: str, access_token: Optional[str] = None, timeout_s: int = 30) -> Any:
    headers = {"User-Agent": "eve-money-button/1.0"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def fetch_character_skills(access_token: str, character_id: int) -> Dict[int, int]:
    # returns mapping skill_type_id -> active_skill_level
    url = f"{ESI_BASE}/characters/{character_id}/skills/?datasource={ESI_DATASOURCE}"
    try:
        j = esi_get(url, access_token=access_token, timeout_s=30)
        skills = j.get("skills") or []
        out: Dict[int, int] = {}
        for s in skills:
            sid = safe_int(s.get("skill_id"), 0)
            lvl = safe_int(s.get("active_skill_level"), 0)
            if sid:
                out[sid] = max(out.get(sid, 0), lvl)
        return out
    except Exception:
        return {}


# -----------------------------
# Price sources
# -----------------------------

def fuzz_price(stats: Dict[str, Any], side: str, mode: str) -> float:
    """
    Convert fuzzwork-like stats into a single 'effective price' number.
    side: "buy" or "sell" (from the orderbook perspective).
    mode:
      - minmax: buy=max, sell=min
      - percentile: use 5% (percentile) when available, else weightedAverage
      - weighted: weightedAverage
    """
    if not stats or side not in stats or not isinstance(stats[side], dict):
        return 0.0
    s = stats[side]
    if mode == "minmax":
        return safe_float(s.get("max" if side == "buy" else "min"), 0.0)
    if mode == "percentile":
        p = safe_float(s.get("percentile"), 0.0)
        if p > 0:
            return p
        return safe_float(s.get("weightedAverage"), 0.0)
    # weighted
    return safe_float(s.get("weightedAverage"), 0.0)


def fuzz_order_count(stats: Dict[str, Any], side: str) -> int:
    if not stats or side not in stats or not isinstance(stats[side], dict):
        return 0
    return safe_int(stats[side].get("orderCount"), 0)


def fuzz_volume(stats: Dict[str, Any], side: str) -> float:
    if not stats or side not in stats or not isinstance(stats[side], dict):
        return 0.0
    return safe_float(stats[side].get("volume"), 0.0)


def download_file(url: str, dest: Path, timeout_s: int = 60) -> None:
    ensure_parent(dest)
    with requests.get(url, stream=True, timeout=timeout_s, headers={"User-Agent": "eve-money-button/1.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_region_prices_from_aggregatecsv(
    cache_csv_path: Path,
    region_ids: Set[int],
    type_filter: Optional[Set[int]] = None,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Load fuzzwork aggregatecsv and keep only requested region_ids, and optionally only type_filter.
    Returns prices[region_id][type_id] -> {"buy": {...}, "sell": {...}}
    """
    ensure_parent(cache_csv_path)
    if not cache_csv_path.exists():
        bz2_path = cache_csv_path.with_suffix(".bz2")
        if not bz2_path.exists():
            print(f"[prices] Downloading {FUZZWORK_AGGREGATECSV_URL}")
            download_file(FUZZWORK_AGGREGATECSV_URL, bz2_path)
        print(f"[prices] Decompressing {bz2_path.name} -> {cache_csv_path.name}")
        import bz2  # stdlib

        with bz2.open(bz2_path, "rb") as src, open(cache_csv_path, "wb") as dst:
            dst.write(src.read())

    prices: Dict[int, Dict[int, Dict[str, Any]]] = {rid: {} for rid in region_ids}
    with open(cache_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = safe_int(row.get("regionID"), 0)
            if rid not in region_ids:
                continue
            tid = safe_int(row.get("typeID"), 0)
            if not tid:
                continue
            if type_filter is not None and tid not in type_filter:
                continue
            side = row.get("buySell")  # "b" or "s"
            if side not in ("b", "s"):
                continue
            side_key = "buy" if side == "b" else "sell"
            entry = prices[rid].setdefault(tid, {"buy": {}, "sell": {}})
            entry[side_key] = {
                "weightedAverage": safe_float(row.get("weightedAvg"), 0.0),
                "max": safe_float(row.get("max"), 0.0),
                "min": safe_float(row.get("min"), 0.0),
                "stddev": safe_float(row.get("stddev"), 0.0),
                "median": safe_float(row.get("median"), 0.0),
                "volume": safe_float(row.get("volume"), 0.0),
                "orderCount": safe_int(row.get("orderCount"), 0),
                "percentile": safe_float(row.get("fivePercent"), 0.0),
            }
    return prices


def fetch_fuzzwork_station_aggregates(station_id: int, type_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping type_id -> {"buy": {...}, "sell": {...}}
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not type_ids:
        return out

    for chunk in chunked(type_ids, STATION_AGG_CHUNK):
        params = {"station": station_id, "types": ",".join(str(x) for x in chunk)}
        try:
            r = requests.get(FUZZWORK_STATION_AGG_URL, params=params, timeout=60, headers={"User-Agent": "eve-money-button/1.0"})
            r.raise_for_status()
            j = r.json()
            # Response keys are strings of type_id
            for k, v in j.items():
                tid = safe_int(k, 0)
                if not tid or not isinstance(v, dict):
                    continue
                # normalize: ensure buy/sell dicts exist if present
                vv: Dict[str, Any] = {"buy": {}, "sell": {}}
                if isinstance(v.get("buy"), dict):
                    vv["buy"] = v["buy"]
                if isinstance(v.get("sell"), dict):
                    vv["sell"] = v["sell"]
                out[tid] = vv
        except Exception:
            # keep going; missing chunks shouldn't kill the whole run
            continue
    return out


def compute_stats_from_orders(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build fuzzwork-like stats dict for a side from a list of orders with fields:
      price, volume_remain
    """
    if not orders:
        return {"weightedAverage": 0.0, "max": 0.0, "min": 0.0, "stddev": 0.0, "median": 0.0, "volume": 0.0, "orderCount": 0, "percentile": 0.0}

    # For weighted average and percentile, weight by remaining volume.
    # Percentile here is "5% price" approximation: weighted average of best 5% of volume.
    prices: List[Tuple[float, float]] = []
    total_vol = 0.0
    for o in orders:
        p = safe_float(o.get("price"), 0.0)
        v = safe_float(o.get("volume_remain") or o.get("volume_remain"), 0.0)
        if p <= 0 or v <= 0:
            continue
        prices.append((p, v))
        total_vol += v
    if not prices or total_vol <= 0:
        return {"weightedAverage": 0.0, "max": 0.0, "min": 0.0, "stddev": 0.0, "median": 0.0, "volume": 0.0, "orderCount": 0, "percentile": 0.0}

    # Sort by price for percentile calc; caller should provide the right direction for side.
    # Here we assume orders passed are already sorted "best first".
    weighted_sum = sum(p * v for p, v in prices)
    wavg = weighted_sum / total_vol

    # median by volume
    cum = 0.0
    median_price = prices[-1][0]
    for p, v in prices:
        cum += v
        if cum >= total_vol / 2.0:
            median_price = p
            break

    # stddev (volume-weighted)
    var = sum(v * ((p - wavg) ** 2) for p, v in prices) / total_vol
    stddev = math.sqrt(var)

    # 5% percentile (best 5% of volume)
    target = total_vol * 0.05
    take = 0.0
    take_sum = 0.0
    for p, v in prices:
        if take >= target:
            break
        dv = min(v, target - take)
        take += dv
        take_sum += p * dv
    percentile = take_sum / take if take > 0 else wavg

    max_p = max(p for p, _ in prices)
    min_p = min(p for p, _ in prices)
    return {
        "weightedAverage": wavg,
        "max": max_p,
        "min": min_p,
        "stddev": stddev,
        "median": median_price,
        "volume": total_vol,
        "orderCount": len(prices),
        "percentile": percentile,
    }


# -----------------------------
# Recipes + SDE-derived type info
# -----------------------------

def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def maybe_build_recipes(force: bool = False) -> None:
    if RECIPES_PATH.exists() and not force:
        return
    print("[update_rankings] recipes.json.gz missing or forced; building from SDE (one-time-ish)…")
    import subprocess

    subprocess.check_call(["python", "scripts/build_recipes.py", "--out", str(RECIPES_PATH)])


def load_recipes() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = read_json_gz(RECIPES_PATH)
    recipes = data.get("recipes") or {}
    types = data.get("types") or {}
    return recipes, types


# -----------------------------
# ESI: adjusted prices + system indices + market data
# -----------------------------

def fetch_adjusted_prices() -> Dict[int, float]:
    url = f"{ESI_BASE}/markets/prices/?datasource={ESI_DATASOURCE}"
    try:
        j = esi_get(url, access_token=None, timeout_s=30)
        out: Dict[int, float] = {}
        for row in j:
            tid = safe_int(row.get("type_id"), 0)
            ap = row.get("adjusted_price")
            if tid and ap is not None:
                out[tid] = safe_float(ap, 0.0)
        return out
    except Exception:
        return {}


def fetch_system_cost_indices(system_id: int) -> Dict[int, float]:
    """
    Returns mapping activity_id -> cost_index for a system.
    Activity IDs:
      1 manufacturing
      11 reactions
      8 invention
    """
    url = f"{ESI_BASE}/industry/systems/?datasource={ESI_DATASOURCE}"
    try:
        j = esi_get(url, access_token=None, timeout_s=60)
        for row in j:
            sid = safe_int(row.get("solar_system_id"), 0)
            if sid != system_id:
                continue
            out: Dict[int, float] = {}
            for ci in row.get("cost_indices") or []:
                aid = safe_int(ci.get("activity_id"), 0)
                val = safe_float(ci.get("cost_index"), 0.0)
                if aid:
                    out[aid] = val
            return out
        return {}
    except Exception:
        return {}


def fetch_market_history(region_id: int, type_id: int) -> List[Dict[str, Any]]:
    url = f"{ESI_BASE}/markets/{region_id}/history/?datasource={ESI_DATASOURCE}&type_id={type_id}"
    try:
        return esi_get(url, access_token=None, timeout_s=30) or []
    except Exception:
        return []


def fetch_region_orders(region_id: int, order_type: str, type_id: int) -> List[Dict[str, Any]]:
    """
    Public regional orders (NPC stations only; structure orders are NOT included).
    order_type: "buy" or "sell"
    """
    orders: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"{ESI_BASE}/markets/{region_id}/orders/"
            f"?datasource={ESI_DATASOURCE}&order_type={order_type}&type_id={type_id}&page={page}"
        )
        try:
            batch = esi_get(url, access_token=None, timeout_s=60)
        except Exception:
            break
        if not batch:
            break
        orders.extend(batch)
        if len(batch) < 1000:
            break
        page += 1
        if page > 50:
            break
        time.sleep(0.12)
    return orders


def fetch_structure_orders(structure_id: int, access_token: str) -> List[Dict[str, Any]]:
    """
    Authenticated structure market orders.
    NOTE: No type filter; returns all orders, paginated.
    """
    orders: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = f"{ESI_BASE}/markets/structures/{structure_id}/?datasource={ESI_DATASOURCE}&page={page}"
        try:
            batch = esi_get(url, access_token=access_token, timeout_s=60)
        except Exception:
            break
        if not batch:
            break
        orders.extend(batch)
        if len(batch) < 1000:
            break
        page += 1
        if page > 200:
            break
        time.sleep(0.12)
    return orders


# -----------------------------
# Industry cost + fee model
# -----------------------------

@dataclass(frozen=True)
class FeeModel:
    sales_tax: float
    broker_fee: float
    facility_tax: float
    structure_job_bonus: float  # multiplier applied to install cost, e.g. 0.98 for -2%


def job_install_cost_per_run(
    adjusted_prices: Dict[int, float],
    system_cost_index: float,
    materials: List[Dict[str, Any]],
    output_qty: float,
    fee: FeeModel,
) -> float:
    """
    Approximate job install cost per run:
      sum(qty * adjusted_price(material)) * system_cost_index * facility_tax * structure_job_bonus
    """
    if system_cost_index <= 0:
        return 0.0
    base = 0.0
    for m in materials:
        tid = safe_int(m.get("type_id"), 0)
        qty = safe_float(m.get("qty"), 0.0)
        ap = adjusted_prices.get(tid, 0.0)
        if tid and qty > 0 and ap > 0:
            base += qty * ap
    base = base / max(output_qty, 1.0)
    return base * system_cost_index * fee.facility_tax * fee.structure_job_bonus


def compute_mode_metrics(
    output_price: float,
    output_qty: float,
    materials: List[Dict[str, Any]],
    time_s: float,
    fee: FeeModel,
    job_cost_per_run: float,
    blueprint_cost: Optional[float],
    blueprint_runs: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Materials: list of {type_id, name, qty, unit_price, unit_m3}
    Returns dict with cost/revenue/fees/profit and material breakdown.
    """
    mat_cost = 0.0
    mats_out: List[Dict[str, Any]] = []
    input_m3 = 0.0
    for m in materials:
        qty = safe_float(m["qty"], 0.0)
        unit_price = safe_float(m["unit_price"], 0.0)
        unit_m3 = safe_float(m.get("unit_m3"), 0.0)
        ext = qty * unit_price
        mat_cost += ext
        input_m3 += qty * unit_m3
        mats_out.append(
            {
                "type_id": m["type_id"],
                "name": m["name"],
                "qty": qty,
                "unit_price": unit_price,
                "extended": ext,
                "unit_m3": unit_m3,
                "extended_m3": qty * unit_m3,
            }
        )

    revenue = output_qty * output_price
    # Market fees (selling). Buying fees ignored for simplicity.
    sales_tax = revenue * fee.sales_tax
    broker_fee = revenue * fee.broker_fee
    sell_fees = sales_tax + broker_fee

    # Blueprint amortization
    bp_cost = 0.0
    if blueprint_cost and blueprint_cost > 0:
        runs = blueprint_runs or 1.0
        bp_cost = blueprint_cost / max(runs, 1.0)

    total_cost = mat_cost + job_cost_per_run + bp_cost + sell_fees
    profit = revenue - total_cost
    roi = profit / total_cost if total_cost > 0 else 0.0
    profit_per_hour = profit / (time_s / 3600.0) if time_s > 0 else 0.0

    return {
        "cost": total_cost,
        "materials_cost": mat_cost,
        "job_cost": job_cost_per_run,
        "blueprint_cost_per_run": bp_cost,
        "revenue": revenue,
        "fees": sell_fees,
        "fee_breakdown": {"sales_tax": sales_tax, "broker_fee": broker_fee},
        "profit": profit,
        "roi": roi,
        "profit_per_hour": profit_per_hour,
        "materials": mats_out,
        "input_m3_per_run": input_m3,
    }


# -----------------------------
# Confidence scoring
# -----------------------------

def compute_confidence_from_market_stats(stats: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Confidence score (0–100) from output stats alone (buy/sell orders).
    Conservative: penalizes low orders, low volume, high spread, high volatility.
    """
    buy = (stats.get("buy") or {}) if isinstance(stats.get("buy"), dict) else {}
    sell = (stats.get("sell") or {}) if isinstance(stats.get("sell"), dict) else {}

    buy_oc = safe_int(buy.get("orderCount"), 0)
    sell_oc = safe_int(sell.get("orderCount"), 0)
    buy_vol = safe_float(buy.get("volume"), 0.0)
    sell_vol = safe_float(sell.get("volume"), 0.0)

    buy_p = safe_float(buy.get("max") or buy.get("weightedAverage") or 0.0, 0.0)
    sell_p = safe_float(sell.get("min") or sell.get("weightedAverage") or 0.0, 0.0)

    spread = max(sell_p - buy_p, 0.0)
    spread_pct = spread / sell_p if sell_p > 0 else 1.0

    buy_std = safe_float(buy.get("stddev"), 0.0)
    sell_std = safe_float(sell.get("stddev"), 0.0)
    vol_pct = 0.0
    if sell_p > 0:
        vol_pct = max(buy_std, sell_std) / sell_p

    score = 100.0

    # Orders and volume
    if buy_oc < 2:
        score -= 35
    elif buy_oc < 5:
        score -= 20
    elif buy_oc < 15:
        score -= 10

    if sell_oc < 2:
        score -= 20
    elif sell_oc < 5:
        score -= 10

    if buy_vol < 10:
        score -= 15
    elif buy_vol < 100:
        score -= 8

    if sell_vol < 10:
        score -= 10
    elif sell_vol < 100:
        score -= 5

    # Spread and volatility
    if spread_pct > 0.8:
        score -= 30
    elif spread_pct > 0.4:
        score -= 18
    elif spread_pct > 0.2:
        score -= 8

    if vol_pct > 1.0:
        score -= 25
    elif vol_pct > 0.5:
        score -= 12
    elif vol_pct > 0.25:
        score -= 6

    score = max(0.0, min(100.0, score))
    details = {
        "buy_order_count": buy_oc,
        "sell_order_count": sell_oc,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "spread_pct": spread_pct,
        "vol_pct": vol_pct,
    }
    return int(round(score)), details


# -----------------------------
# Depth simulation (orderbook)
# -----------------------------

@dataclass
class OrderBook:
    side: str  # "buy" or "sell"
    orders: List[Tuple[float, float]]  # [(price, volume_remain)] sorted best-first

    @property
    def best(self) -> float:
        return self.orders[0][0] if self.orders else 0.0


def build_order_book_from_esi(orders: List[Dict[str, Any]], side: str) -> OrderBook:
    # ESI orders fields: price, volume_remain, is_buy_order, location_id
    ob: List[Tuple[float, float]] = []
    for o in orders:
        p = safe_float(o.get("price"), 0.0)
        v = safe_float(o.get("volume_remain"), 0.0)
        if p <= 0 or v <= 0:
            continue
        ob.append((p, v))
    # Sort best-first:
    if side == "sell":
        ob.sort(key=lambda x: x[0])  # cheapest first
    else:
        ob.sort(key=lambda x: -x[0])  # highest first
    return OrderBook(side=side, orders=ob)


def max_units_with_slippage(book: OrderBook, allowed_slippage: float) -> Tuple[float, float]:
    """
    Returns (max_units, avg_price) you can fill before average price exceeds best*(1+slippage) for sells
    or drops below best*(1-slippage) for buys.
    """
    if not book.orders:
        return 0.0, 0.0
    best = book.best
    if best <= 0:
        return 0.0, 0.0

    if book.side == "sell":
        threshold = best * (1.0 + allowed_slippage)
        cmp_ok = lambda avg: avg <= threshold
    else:
        threshold = best * (1.0 - allowed_slippage)
        cmp_ok = lambda avg: avg >= threshold

    total_units = 0.0
    total_value = 0.0
    max_ok_units = 0.0
    max_ok_avg = 0.0

    for price, vol in book.orders:
        if vol <= 0:
            continue
        total_units += vol
        total_value += price * vol
        avg = total_value / total_units
        if cmp_ok(avg):
            max_ok_units = total_units
            max_ok_avg = avg
        else:
            break

    return max_ok_units, max_ok_avg


def take_units(book: OrderBook, units: float) -> Tuple[float, float, float]:
    """
    Take up to `units` from the book (best-first).
    Returns (filled_units, total_value, avg_price).
    """
    need = units
    filled = 0.0
    total_value = 0.0
    for price, vol in book.orders:
        if need <= 0:
            break
        dv = min(vol, need)
        filled += dv
        total_value += dv * price
        need -= dv
    avg = total_value / filled if filled > 0 else 0.0
    return filled, total_value, avg


def fetch_orderbook_for_market(
    market: Market,
    order_type: str,  # "buy" or "sell"
    type_id: int,
    access_token: Optional[str],
    _structure_cache: Dict[int, List[Dict[str, Any]]],
) -> OrderBook:
    """
    Fetch orderbook for a single type in a market.
    For stations/regions: use regional orders and optionally filter to station.
    For structures: uses structure orders (cached per structure).
    """
    if market.kind == "structure":
        if not access_token:
            return OrderBook(side=order_type, orders=[])
        sid = market.structure_id or 0
        if sid <= 0:
            return OrderBook(side=order_type, orders=[])
        if sid not in _structure_cache:
            print(f"[depth] Fetching structure orders for {market.name} ({sid}) …")
            _structure_cache[sid] = fetch_structure_orders(sid, access_token)
        orders_all = _structure_cache[sid]
        filtered = []
        want_buy = (order_type == "buy")
        for o in orders_all:
            if safe_int(o.get("type_id"), 0) != type_id:
                continue
            if bool(o.get("is_buy_order")) != want_buy:
                continue
            filtered.append(o)
        return build_order_book_from_esi(filtered, side=order_type)

    # station / region (NPC orders)
    if market.region_id is None:
        return OrderBook(side=order_type, orders=[])
    raw = fetch_region_orders(market.region_id, order_type=order_type, type_id=type_id)
    if market.kind == "station" and market.station_id:
        raw = [o for o in raw if safe_int(o.get("location_id"), 0) == market.station_id]
    return build_order_book_from_esi(raw, side=order_type)


def validate_depth_for_recipe(
    input_market: Market,
    output_market: Market,
    product_type_id: int,
    output_qty: float,
    materials: List[Dict[str, Any]],  # {type_id, qty, unit_price}
    input_slippage: float,
    output_slippage: float,
    depth_safety: float,
    max_materials: int,
    access_token: Optional[str],
    structure_cache: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Depth validate using real orderbooks.
    Only considers up to max_materials most expensive materials (by extended cost) for input depth.
    """
    # pick top materials by cost share
    mats_sorted = sorted(
        materials,
        key=lambda m: safe_float(m.get("qty"), 0.0) * safe_float(m.get("unit_price"), 0.0),
        reverse=True,
    )
    depth_mats = mats_sorted[: max_materials if max_materials > 0 else len(mats_sorted)]

    max_runs_in = float("inf")
    input_books: List[Dict[str, Any]] = []

    for m in depth_mats:
        tid = safe_int(m.get("type_id"), 0)
        qty_per = safe_float(m.get("qty"), 0.0)
        if tid <= 0 or qty_per <= 0:
            continue
        ob = fetch_orderbook_for_market(input_market, "sell", tid, access_token, structure_cache)
        max_units, avg = max_units_with_slippage(ob, input_slippage)
        max_runs = math.floor(max_units / qty_per) if qty_per > 0 else 0
        max_runs_in = min(max_runs_in, max_runs)
        input_books.append(
            {
                "type_id": tid,
                "qty_per_run": qty_per,
                "max_units": max_units,
                "max_runs": max_runs,
                "avg_price_to_slippage": avg,
                "best": ob.best,
                "orders_considered": len(ob.orders),
            }
        )

    if max_runs_in == float("inf"):
        max_runs_in = 0

    out_book = fetch_orderbook_for_market(output_market, "buy", product_type_id, access_token, structure_cache)
    max_units_out, avg_out = max_units_with_slippage(out_book, output_slippage)
    max_runs_out = math.floor(max_units_out / max(output_qty, 1.0))

    recommended_runs = math.floor(min(max_runs_in, max_runs_out) * depth_safety)
    recommended_runs = max(recommended_runs, 0)

    # Expected numbers at recommended runs (use actual fill from orderbooks)
    needed_out_units = recommended_runs * output_qty
    filled_out, out_value, out_avg = take_units(out_book, needed_out_units)

    needed_inputs = []
    in_total_value = 0.0
    in_filled_ok = True
    for m in depth_mats:
        tid = safe_int(m.get("type_id"), 0)
        qty_per = safe_float(m.get("qty"), 0.0)
        need_units = recommended_runs * qty_per
        ob = fetch_orderbook_for_market(input_market, "sell", tid, access_token, structure_cache)
        filled, value, avg = take_units(ob, need_units)
        if filled + 1e-9 < need_units:
            in_filled_ok = False
        in_total_value += value
        needed_inputs.append(
            {
                "type_id": tid,
                "need_units": need_units,
                "filled_units": filled,
                "avg_price": avg,
            }
        )

    out_filled_ok = (filled_out + 1e-9 >= needed_out_units)

    return {
        "max_runs_input": int(max_runs_in),
        "max_runs_output": int(max_runs_out),
        "recommended_runs": int(recommended_runs),
        "input": {
            "market": input_market.key,
            "materials": input_books,
            "filled_ok": in_filled_ok,
            "total_value": in_total_value,
        },
        "output": {
            "market": output_market.key,
            "filled_ok": out_filled_ok,
            "best": out_book.best,
            "avg_price_to_slippage": avg_out,
            "total_value": out_value,
            "avg_price": out_avg,
            "orders_considered": len(out_book.orders),
        },
        "expected": {
            "runs": int(recommended_runs),
            "output_units": needed_out_units,
            "inputs_value_depth_mats": in_total_value,
            "output_value": out_value,
        },
    }


# -----------------------------
# Skill-aware time modifiers (optional)
# -----------------------------

def build_name_to_type_id(types: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tid_str, t in types.items():
        try:
            tid = int(tid_str)
        except Exception:
            continue
        name = t.get("name")
        if isinstance(name, str) and name:
            # Keep first seen; skill names are unique anyway
            out.setdefault(name, tid)
    return out


def skill_level(skills_by_id: Dict[int, int], name_to_id: Dict[str, int], skill_name: str) -> int:
    tid = name_to_id.get(skill_name)
    if not tid:
        return 0
    return skills_by_id.get(tid, 0)


def apply_time_modifiers(base_time_s: float, category: str, skill_levels: Dict[str, int]) -> float:
    """
    category: manufacturing|reactions|invention
    """
    t = base_time_s
    if category in ("manufacturing", "invention", "t2"):
        ind = skill_levels.get("Industry", 0)
        adv = skill_levels.get("Advanced Industry", 0)
        # Industry: -4% per level, Advanced Industry: -3% per level (best-effort; game values may change)
        t *= (1.0 - 0.04 * ind)
        t *= (1.0 - 0.03 * adv)
    if category in ("reactions",):
        rx = skill_levels.get("Reactions", 0)
        t *= (1.0 - 0.04 * rx)
    return max(t, 1.0)


# -----------------------------
# Time to liquidate (history-based)
# -----------------------------

def avg_daily_volume(history: List[Dict[str, Any]], days: int = 7) -> float:
    # history is list of {"date": "...", "volume": ..., ...} daily bars
    if not history:
        return 0.0
    # take last N entries
    h = history[-days:]
    vols = [safe_float(x.get("volume"), 0.0) for x in h if safe_float(x.get("volume"), 0.0) > 0]
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def ttl_bucket(hours: float) -> str:
    if hours <= 0 or math.isinf(hours) or math.isnan(hours):
        return "unknown"
    if hours <= 12:
        return "<12h"
    if hours <= 24:
        return "<24h"
    if hours <= 72:
        return "<72h"
    return ">72h"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    # Back-compat (single market mode)
    parser.add_argument("--region-id", type=int, default=10000002, help="(legacy) region id for single-market mode")
    parser.add_argument("--station-id", type=int, default=60003760, help="(legacy) station id for single-market mode")

    # Multi-market mode
    parser.add_argument("--markets-config", type=str, default="data/markets.json", help="Path to markets config JSON")
    parser.add_argument("--buy-market", type=str, default=None, help="Market key to price INPUTS (materials/BPO)")
    parser.add_argument("--sell-markets", type=str, default=None, help="Comma-separated market keys to consider for OUTPUTS")

    parser.add_argument("--price-mode", choices=["minmax", "percentile", "weighted"], default="percentile")
    parser.add_argument("--pricing-scope", choices=["region", "station"], default="station")

    parser.add_argument("--min-output-buy-orders", type=int, default=3)
    parser.add_argument("--min-blueprint-sell-orders", type=int, default=1)
    parser.add_argument("--min-job-time-s", type=int, default=600)
    parser.add_argument("--max-rows", type=int, default=400)

    # Refining liquidity thresholds (separate)
    parser.add_argument("--min-ref-input-sell-orders", type=int, default=5)
    parser.add_argument("--min-ref-output-buy-orders", type=int, default=3)

    # Confidence + depth validation
    parser.add_argument("--confidence-history", action="store_true", help="Use ESI market history to bump confidence / TTL")
    parser.add_argument("--depth-top-n", type=int, default=80)
    parser.add_argument("--depth-max-materials", type=int, default=8)
    parser.add_argument("--input-slippage", type=float, default=0.03)
    parser.add_argument("--output-slippage", type=float, default=0.03)
    parser.add_argument("--depth-safety", type=float, default=0.85)
    parser.add_argument("--best-min-confidence", type=int, default=50, help="When choosing best sell market, ignore markets below this confidence")

    # Fees / costs
    parser.add_argument("--sales-tax", type=float, default=0.036)      # ~3.6% (example; adjust)
    parser.add_argument("--broker-fee", type=float, default=0.03)      # ~3.0% (example; adjust)
    parser.add_argument("--facility-tax", type=float, default=1.0)     # multiplier on job install cost
    parser.add_argument("--structure-job-bonus", type=float, default=1.0)  # multiplier on job install cost

    # Build locations (for system cost indices)
    parser.add_argument("--mfg-system-id", type=int, default=30000142, help="System where you manufacture (for cost index)")
    parser.add_argument("--rx-system-id", type=int, default=30000142, help="System where you run reactions (for cost index)")
    parser.add_argument("--inv-system-id", type=int, default=30000142, help="System where you invent (for cost index)")

    parser.add_argument("--force-recipes", action="store_true")
    parser.add_argument("--out", type=str, default=str(OUT_PATH_DEFAULT))

    args = parser.parse_args()

    # Build recipes if needed
    maybe_build_recipes(force=args.force_recipes)
    recipes, types = load_recipes()
    name_to_tid = build_name_to_type_id(types)

    # SSO (optional)
    access_token = None
    character: Dict[str, Any] = {}
    sso_client_id = os.environ.get("EVE_SSO_CLIENT_ID")
    sso_client_secret = os.environ.get("EVE_SSO_CLIENT_SECRET")
    sso_refresh = os.environ.get("EVE_SSO_REFRESH_TOKEN")
    if sso_client_id and sso_client_secret and sso_refresh:
        access_token = sso_access_token_from_refresh(sso_client_id, sso_client_secret, sso_refresh)
        if access_token:
            cid, cname = fetch_character_id_and_name(access_token)
            if cid and cname:
                skills_by_id = fetch_character_skills(access_token, cid)
                # Store a small human-friendly subset (best effort)
                skill_levels = {
                    "Industry": skill_level(skills_by_id, name_to_tid, "Industry"),
                    "Advanced Industry": skill_level(skills_by_id, name_to_tid, "Advanced Industry"),
                    "Reactions": skill_level(skills_by_id, name_to_tid, "Reactions"),
                }
                character = {"id": cid, "name": cname, "skills": skill_levels}
            else:
                character = {"id": None, "name": None}

    # Markets configuration
    markets_path = Path(args.markets_config)
    if markets_path.exists():
        cfg = load_markets_config(markets_path)
    else:
        # fallback: legacy single-market config
        cfg = MarketsConfig(
            markets={
                "legacy": Market(
                    key="legacy",
                    name=f"Region {args.region_id}",
                    kind="station" if args.pricing_scope == "station" else "region",
                    region_id=args.region_id,
                    station_id=args.station_id if args.pricing_scope == "station" else None,
                )
            },
            default_buy="legacy",
            default_sells=["legacy"],
        )

    buy_key = args.buy_market or cfg.default_buy
    sell_keys = parse_csv_list(args.sell_markets) or cfg.default_sells or [buy_key]

    # sanitize keys
    if buy_key not in cfg.markets:
        raise SystemExit(f"Unknown buy market key: {buy_key}")
    sell_keys = [k for k in sell_keys if k in cfg.markets and k != ""]
    if not sell_keys:
        sell_keys = [buy_key]

    buy_market = cfg.markets[buy_key]
    sell_markets = [cfg.markets[k] for k in sell_keys]

    # Drop invalid markets (e.g. structure markets without structure_id)
    sell_markets = [m for m in sell_markets if m.is_valid()]
    if not sell_markets:
        sell_markets = [buy_market] if buy_market.is_valid() else []

    if not buy_market.is_valid():
        # In the worst case, fall back to legacy region
        buy_market = Market(key="legacy", name=f"Region {args.region_id}", kind="region", region_id=args.region_id)

    print(f"[markets] Buy market: {buy_market.key} ({buy_market.name})")
    print("[markets] Sell markets:", ", ".join(f"{m.key}({m.name})" for m in sell_markets))

    # Build type id sets needed for pricing
    mfg_recipes = recipes.get("manufacturing") or []
    rx_recipes = recipes.get("reactions") or []
    ref_recipes = recipes.get("refining") or []
    inv_recipes = recipes.get("invention") or []

    buy_type_ids: Set[int] = set()
    sell_type_ids: Set[int] = set()

    # Manufacturing / reactions inputs and blueprints from buy market; outputs from sell markets
    for r in mfg_recipes:
        buy_type_ids.add(safe_int(r.get("blueprint_type_id"), 0))
        sell_type_ids.add(safe_int(r.get("product_type_id"), 0))
        for m in r.get("materials") or []:
            buy_type_ids.add(safe_int(m.get("type_id"), 0))

    for r in rx_recipes:
        buy_type_ids.add(safe_int(r.get("blueprint_type_id"), 0))
        sell_type_ids.add(safe_int(r.get("product_type_id"), 0))
        for m in r.get("materials") or []:
            buy_type_ids.add(safe_int(m.get("type_id"), 0))

    # Refining: ore/ice inputs from buy market, mineral outputs valued in sell markets
    for r in ref_recipes:
        buy_type_ids.add(safe_int(r.get("input_type_id"), 0))
        for o in r.get("outputs") or []:
            sell_type_ids.add(safe_int(o.get("type_id"), 0))

    # Invention: datacores/decryptors and T2 build inputs are bought in buy market; outputs sold in sell markets
    for r in inv_recipes:
        # invention consumes these
        for m in r.get("invention_materials") or []:
            buy_type_ids.add(safe_int(m.get("type_id"), 0))
        # produced blueprint then manufactured; revenue from final product
        sell_type_ids.add(safe_int(r.get("product_type_id"), 0))
        # and manufacturing materials for the T2 product if embedded
        for m in r.get("manufacturing_materials") or []:
            buy_type_ids.add(safe_int(m.get("type_id"), 0))

    buy_type_ids.discard(0)
    sell_type_ids.discard(0)

    # Load pricing for buy market and sell markets
    market_prices: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # Station scope: use station aggregates for station markets; Region scope: aggregatecsv.
    # We always load at least buy market.
    def load_market_prices_for(market: Market, needed: Set[int]) -> Dict[int, Dict[str, Any]]:
        needed_list = sorted(needed)
        if market.kind == "station" and args.pricing_scope == "station" and market.station_id:
            return fetch_fuzzwork_station_aggregates(market.station_id, needed_list)
        if market.kind == "region" or args.pricing_scope == "region":
            rid = market.region_id or args.region_id
            regions = {rid}
            prices_by_region = load_region_prices_from_aggregatecsv(CACHE_DIR / "aggregatecsv.csv", regions, type_filter=needed)
            return prices_by_region.get(rid, {})
        # structure: we'll compute stats from structure orders if token is available
        if market.kind == "structure" and market.structure_id and access_token:
            orders = fetch_structure_orders(market.structure_id, access_token)
            by_type_buy: Dict[int, List[Dict[str, Any]]] = {}
            by_type_sell: Dict[int, List[Dict[str, Any]]] = {}
            for o in orders:
                tid = safe_int(o.get("type_id"), 0)
                if tid not in needed:
                    continue
                if bool(o.get("is_buy_order")):
                    by_type_buy.setdefault(tid, []).append(o)
                else:
                    by_type_sell.setdefault(tid, []).append(o)
            stats: Dict[int, Dict[str, Any]] = {}
            for tid in needed:
                b = by_type_buy.get(tid, [])
                s = by_type_sell.get(tid, [])
                # sort for percentile correctness
                b_sorted = sorted(b, key=lambda x: -safe_float(x.get("price"), 0.0))
                s_sorted = sorted(s, key=lambda x: safe_float(x.get("price"), 0.0))
                stats[tid] = {
                    "buy": compute_stats_from_orders(
                        [{"price": o.get("price"), "volume_remain": o.get("volume_remain")} for o in b_sorted]
                    ),
                    "sell": compute_stats_from_orders(
                        [{"price": o.get("price"), "volume_remain": o.get("volume_remain")} for o in s_sorted]
                    ),
                }
            return stats
        return {}

    # buy market needs buy_type_ids (and may also be a sell market)
    market_prices[buy_market.key] = load_market_prices_for(buy_market, buy_type_ids.union(sell_type_ids if buy_market in sell_markets else set()))

    # sell markets need only sell_type_ids (plus maybe blueprint? not needed)
    for m in sell_markets:
        if m.key == buy_market.key:
            continue
        market_prices[m.key] = load_market_prices_for(m, sell_type_ids)

    # ESI adjusted prices + system cost indices
    adjusted_prices = fetch_adjusted_prices()
    mfg_ci = fetch_system_cost_indices(args.mfg_system_id).get(1, 0.0)
    rx_ci = fetch_system_cost_indices(args.rx_system_id).get(11, 0.0)
    inv_ci = fetch_system_cost_indices(args.inv_system_id).get(8, 0.0)

    fee_model = FeeModel(
        sales_tax=args.sales_tax,
        broker_fee=args.broker_fee,
        facility_tax=args.facility_tax,
        structure_job_bonus=args.structure_job_bonus,
    )

    # Helper functions for type names & volume
    def tname(tid: int) -> str:
        return (types.get(str(tid)) or {}).get("name") or f"type:{tid}"

    def tvol(tid: int) -> float:
        return safe_float((types.get(str(tid)) or {}).get("volume"), 0.0)

    def stats_for(market_key: str, tid: int) -> Dict[str, Any]:
        return market_prices.get(market_key, {}).get(tid, {})

    def price_for(market_key: str, tid: int, side: str) -> float:
        return fuzz_price(stats_for(market_key, tid), side, args.price_mode)

    def summarize_alternatives(per_market: Dict[str, Any], mode: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Small, JSON-friendly summary of best destinations by mode."""
        items = sorted(
            per_market.items(),
            key=lambda kv: safe_float((kv[1].get(mode) or {}).get("profit_per_hour"), 0.0),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for k, v in items[:top_k]:
            m = v.get(mode) or {}
            out.append(
                {
                    "market": k,
                    "market_name": v.get("market_name") or k,
                    "profit_per_hour": safe_float(m.get("profit_per_hour"), 0.0),
                    "profit": safe_float(m.get("profit"), 0.0),
                    "roi": safe_float(m.get("roi"), 0.0),
                    "confidence": safe_int(v.get("confidence"), 0),
                }
            )
        return out


    # -----------------------------
    # Build rows
    # -----------------------------

    manufacturing_rows: List[Dict[str, Any]] = []
    reaction_rows: List[Dict[str, Any]] = []
    refining_rows: List[Dict[str, Any]] = []
    t2_rows: List[Dict[str, Any]] = []

    # Skill-aware time multipliers
    skill_levels = (character.get("skills") or {}) if isinstance(character.get("skills"), dict) else {}

    # ---------- Manufacturing ----------
    for r in mfg_recipes:
        pt = safe_int(r.get("product_type_id"), 0)
        bp = safe_int(r.get("blueprint_type_id"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0) if r.get("output_qty") is not None else 1.0
        time_s = safe_float(r.get("time_s"), 0.0)
        if time_s < args.min_job_time_s:
            continue
        time_s = apply_time_modifiers(time_s, "manufacturing", skill_levels)

        # blueprint price availability check (in buy market)
        bp_stats = stats_for(buy_market.key, bp)
        if fuzz_order_count(bp_stats, "sell") < args.min_blueprint_sell_orders:
            continue
        bp_cost = price_for(buy_market.key, bp, "sell")
        if bp_cost <= 0:
            continue

        # materials in buy market
        mats_instant: List[Dict[str, Any]] = []
        mats_patient: List[Dict[str, Any]] = []
        mats_for_depth: List[Dict[str, Any]] = []

        ok = True
        for m in r.get("materials") or []:
            tid = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if tid <= 0 or qty <= 0:
                continue
            st = stats_for(buy_market.key, tid)
            sellp = fuzz_price(st, "sell", args.price_mode)  # instant buy from sells
            buyp = fuzz_price(st, "buy", args.price_mode)   # patient buy via buys
            if sellp <= 0 or buyp <= 0:
                ok = False
                break
            mats_instant.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": sellp, "unit_m3": tvol(tid)})
            mats_patient.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": buyp, "unit_m3": tvol(tid)})
            mats_for_depth.append({"type_id": tid, "qty": qty, "unit_price": sellp})
        if not ok:
            continue

        # job cost per run (manufacturing activity)
        job_cost = job_install_cost_per_run(adjusted_prices, mfg_ci, mats_for_depth, out_qty, fee_model)

        # Evaluate across sell markets
        per_market: Dict[str, Any] = {}
        best_instant_key = None
        best_patient_key = None
        best_instant_profitph = -1e99
        best_patient_profitph = -1e99

        for sm in sell_markets:
            # output prices in this sell market
            out_buy = price_for(sm.key, pt, "buy")   # instant sell to buys
            out_sell = price_for(sm.key, pt, "sell") # patient sell via sells
            if out_buy <= 0 or out_sell <= 0:
                continue

            inst = compute_mode_metrics(out_buy, out_qty, mats_instant, time_s, fee_model, job_cost, blueprint_cost=bp_cost, blueprint_runs=r.get("blueprint_runs"))
            pat = compute_mode_metrics(out_sell, out_qty, mats_patient, time_s, fee_model, job_cost, blueprint_cost=bp_cost, blueprint_runs=r.get("blueprint_runs"))

            conf, conf_details = compute_confidence_from_market_stats(stats_for(sm.key, pt))
            per_market[sm.key] = {
                "market_name": sm.name,
                "confidence": conf,
                "confidence_details": conf_details,
                "instant": inst,
                "patient": pat,
            }

            if conf >= args.best_min_confidence and fuzz_order_count(stats_for(sm.key, pt), "buy") >= args.min_output_buy_orders:
                if inst["profit_per_hour"] > best_instant_profitph:
                    best_instant_profitph = inst["profit_per_hour"]
                    best_instant_key = sm.key
                if pat["profit_per_hour"] > best_patient_profitph:
                    best_patient_profitph = pat["profit_per_hour"]
                    best_patient_key = sm.key

        if not per_market:
            continue

        # If no market met confidence threshold, fall back to pure best profit
        if best_instant_key is None:
            best_instant_key = max(per_market.keys(), key=lambda k: per_market[k]["instant"]["profit_per_hour"])
        if best_patient_key is None:
            best_patient_key = max(per_market.keys(), key=lambda k: per_market[k]["patient"]["profit_per_hour"])

        best_inst = per_market[best_instant_key]["instant"]
        best_pat = per_market[best_patient_key]["patient"]

        # hauling metrics (per run) based on best instant metrics (materials same across markets)
        output_m3 = out_qty * tvol(pt)
        total_m3 = best_inst.get("input_m3_per_run", 0.0) + output_m3
        profit_per_m3_total = best_inst["profit"] / total_m3 if total_m3 > 0 else 0.0
        profit_per_m3_out = best_inst["profit"] / output_m3 if output_m3 > 0 else 0.0

        manufacturing_rows.append(
            {
                "category": "manufacturing",
                "product_type_id": pt,
                "product_name": tname(pt),
                "blueprint_type_id": bp,
                "blueprint_name": tname(bp),
                "output_qty": out_qty,
                "time_s": time_s,
                "blueprint_cost": bp_cost,
                "blueprint_runs": r.get("blueprint_runs"),
                "blueprint_sell_orders": fuzz_order_count(bp_stats, "sell"),
                "payback_runs": (int(math.ceil(bp_cost / best_inst["profit"])) if best_inst.get("profit", 0.0) > 0 else None),
                "alternatives": {
                    "instant": summarize_alternatives(per_market, "instant", top_k=3),
                    "patient": summarize_alternatives(per_market, "patient", top_k=3),
                },  # for UI transparency
                "best_market": {"instant": best_instant_key, "patient": best_patient_key},
                "best_market_name": {"instant": cfg.markets[best_instant_key].name if best_instant_key in cfg.markets else best_instant_key,
                                     "patient": cfg.markets[best_patient_key].name if best_patient_key in cfg.markets else best_patient_key},
                "instant": best_inst,
                "patient": best_pat,
                "confidence": per_market[best_instant_key]["confidence"],
                "hauling": {
                    "input_m3_per_run": best_inst.get("input_m3_per_run", 0.0),
                    "output_m3_per_run": output_m3,
                    "total_m3_per_run": total_m3,
                    "profit_per_m3_total": profit_per_m3_total,
                    "profit_per_m3_out": profit_per_m3_out,
                },
                "depth": None,
                "ttl": None,
            }
        )

    # ---------- Reactions ----------
    for r in rx_recipes:
        pt = safe_int(r.get("product_type_id"), 0)
        bp = safe_int(r.get("blueprint_type_id"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0) if r.get("output_qty") is not None else 1.0
        time_s = safe_float(r.get("time_s"), 0.0)
        if time_s < args.min_job_time_s:
            continue
        time_s = apply_time_modifiers(time_s, "reactions", skill_levels)

        bp_stats = stats_for(buy_market.key, bp)
        if fuzz_order_count(bp_stats, "sell") < args.min_blueprint_sell_orders:
            continue
        bp_cost = price_for(buy_market.key, bp, "sell")
        if bp_cost <= 0:
            continue

        mats_instant: List[Dict[str, Any]] = []
        mats_patient: List[Dict[str, Any]] = []
        mats_for_depth: List[Dict[str, Any]] = []
        ok = True
        for m in r.get("materials") or []:
            tid = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if tid <= 0 or qty <= 0:
                continue
            st = stats_for(buy_market.key, tid)
            sellp = fuzz_price(st, "sell", args.price_mode)
            buyp = fuzz_price(st, "buy", args.price_mode)
            if sellp <= 0 or buyp <= 0:
                ok = False
                break
            mats_instant.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": sellp, "unit_m3": tvol(tid)})
            mats_patient.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": buyp, "unit_m3": tvol(tid)})
            mats_for_depth.append({"type_id": tid, "qty": qty, "unit_price": sellp})
        if not ok:
            continue

        job_cost = job_install_cost_per_run(adjusted_prices, rx_ci, mats_for_depth, out_qty, fee_model)

        per_market: Dict[str, Any] = {}
        best_instant_key = None
        best_patient_key = None
        best_instant_profitph = -1e99
        best_patient_profitph = -1e99

        for sm in sell_markets:
            out_buy = price_for(sm.key, pt, "buy")
            out_sell = price_for(sm.key, pt, "sell")
            if out_buy <= 0 or out_sell <= 0:
                continue

            inst = compute_mode_metrics(out_buy, out_qty, mats_instant, time_s, fee_model, job_cost, blueprint_cost=bp_cost, blueprint_runs=r.get("blueprint_runs"))
            pat = compute_mode_metrics(out_sell, out_qty, mats_patient, time_s, fee_model, job_cost, blueprint_cost=bp_cost, blueprint_runs=r.get("blueprint_runs"))

            conf, conf_details = compute_confidence_from_market_stats(stats_for(sm.key, pt))
            per_market[sm.key] = {
                "market_name": sm.name,
                "confidence": conf,
                "confidence_details": conf_details,
                "instant": inst,
                "patient": pat,
            }
            if conf >= args.best_min_confidence and fuzz_order_count(stats_for(sm.key, pt), "buy") >= args.min_output_buy_orders:
                if inst["profit_per_hour"] > best_instant_profitph:
                    best_instant_profitph = inst["profit_per_hour"]
                    best_instant_key = sm.key
                if pat["profit_per_hour"] > best_patient_profitph:
                    best_patient_profitph = pat["profit_per_hour"]
                    best_patient_key = sm.key

        if not per_market:
            continue
        if best_instant_key is None:
            best_instant_key = max(per_market.keys(), key=lambda k: per_market[k]["instant"]["profit_per_hour"])
        if best_patient_key is None:
            best_patient_key = max(per_market.keys(), key=lambda k: per_market[k]["patient"]["profit_per_hour"])

        best_inst = per_market[best_instant_key]["instant"]
        best_pat = per_market[best_patient_key]["patient"]

        output_m3 = out_qty * tvol(pt)
        total_m3 = best_inst.get("input_m3_per_run", 0.0) + output_m3
        profit_per_m3_total = best_inst["profit"] / total_m3 if total_m3 > 0 else 0.0

        reaction_rows.append(
            {
                "category": "reactions",
                "product_type_id": pt,
                "product_name": tname(pt),
                "blueprint_type_id": bp,
                "blueprint_name": tname(bp),
                "output_qty": out_qty,
                "time_s": time_s,
                "blueprint_cost": bp_cost,
                "blueprint_runs": r.get("blueprint_runs"),
                "blueprint_sell_orders": fuzz_order_count(bp_stats, "sell"),
                "payback_runs": (int(math.ceil(bp_cost / best_inst["profit"])) if best_inst.get("profit", 0.0) > 0 else None),
                "alternatives": {
                    "instant": summarize_alternatives(per_market, "instant", top_k=3),
                    "patient": summarize_alternatives(per_market, "patient", top_k=3),
                },
                "best_market": {"instant": best_instant_key, "patient": best_patient_key},
                "best_market_name": {"instant": cfg.markets[best_instant_key].name if best_instant_key in cfg.markets else best_instant_key,
                                     "patient": cfg.markets[best_patient_key].name if best_patient_key in cfg.markets else best_patient_key},
                "instant": best_inst,
                "patient": best_pat,
                "confidence": per_market[best_instant_key]["confidence"],
                "hauling": {
                    "input_m3_per_run": best_inst.get("input_m3_per_run", 0.0),
                    "output_m3_per_run": output_m3,
                    "total_m3_per_run": total_m3,
                    "profit_per_m3_total": profit_per_m3_total,
                },
                "depth": None,
                "ttl": None,
            }
        )

    # ---------- Refining ----------
    # Refining rows stay simpler (no instant/patient). We treat:
    #  - buy ore/ice in buy market (sell price)
    #  - sell minerals basket into BUY orders of best sell market
    for r in ref_recipes:
        in_tid = safe_int(r.get("input_type_id"), 0)
        units = safe_float(r.get("batch_units"), 0.0)
        batch_m3 = safe_float(r.get("batch_m3"), 0.0)
        if in_tid <= 0 or units <= 0:
            continue
        st_in = stats_for(buy_market.key, in_tid)
        if fuzz_order_count(st_in, "sell") < args.min_ref_input_sell_orders:
            continue
        in_unit_price = fuzz_price(st_in, "sell", args.price_mode)
        if in_unit_price <= 0:
            continue
        cost = in_unit_price * units

        # value outputs per sell market
        best_key = None
        best_profit = -1e99
        best_rev = 0.0
        best_conf = 0

        per_market_val: Dict[str, Any] = {}
        for sm in sell_markets:
            rev = 0.0
            ok = True
            worst_buy_oc = 1_000_000
            for o in r.get("outputs") or []:
                otid = safe_int(o.get("type_id"), 0)
                oqty = safe_float(o.get("qty"), 0.0)
                if otid <= 0 or oqty <= 0:
                    continue
                st = stats_for(sm.key, otid)
                if fuzz_order_count(st, "buy") < args.min_ref_output_buy_orders:
                    ok = False
                    break
                p = fuzz_price(st, "buy", args.price_mode)
                if p <= 0:
                    ok = False
                    break
                rev += p * oqty
                worst_buy_oc = min(worst_buy_oc, fuzz_order_count(st, "buy"))
            if not ok:
                continue
            profit = rev - cost
            roi = profit / cost if cost > 0 else 0.0
            conf, _ = compute_confidence_from_market_stats(stats_for(sm.key, in_tid))
            per_market_val[sm.key] = {"revenue": rev, "profit": profit, "roi": roi, "confidence": conf}
            if conf >= args.best_min_confidence and profit > best_profit:
                best_profit = profit
                best_rev = rev
                best_key = sm.key
                best_conf = conf

        if best_key is None and per_market_val:
            best_key = max(per_market_val.keys(), key=lambda k: per_market_val[k]["profit"])
            best_profit = per_market_val[best_key]["profit"]
            best_rev = per_market_val[best_key]["revenue"]
            best_conf = per_market_val[best_key]["confidence"]

        if best_key is None:
            continue

        refining_rows.append(
            {
                "category": "refining",
                "input_type_id": in_tid,
                "input_name": tname(in_tid),
                "batch_units": units,
                "batch_m3": batch_m3,
                "cost": cost,
                "revenue": best_rev,
                "profit": best_profit,
                "roi": (best_profit / cost) if cost > 0 else 0.0,
                "profit_per_m3": best_profit / batch_m3 if batch_m3 > 0 else 0.0,
                "best_market": best_key,
                "best_market_name": cfg.markets[best_key].name if best_key in cfg.markets else best_key,
                "confidence": best_conf,
                "outputs": [
                    {"type_id": safe_int(o.get("type_id"), 0), "name": tname(safe_int(o.get("type_id"), 0)), "qty": safe_float(o.get("qty"), 0.0)}
                    for o in r.get("outputs") or []
                    if safe_int(o.get("type_id"), 0) > 0 and safe_float(o.get("qty"), 0.0) > 0
                ],
                "per_market": per_market_val,
            }
        )

    # ---------- T2 / Invention pipeline (best-effort) ----------
    # We assume inv_recipes items already include:
    #  - invention_materials: [{type_id, qty}]
    #  - manufacturing_materials: [{type_id, qty}]
    #  - product_type_id, output_qty, time_s, success_chance, attempts_per_success (or derived)
    for r in inv_recipes:
        pt = safe_int(r.get("product_type_id"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0) if r.get("output_qty") is not None else 1.0
        time_s = safe_float(r.get("time_s"), 0.0)
        if time_s < args.min_job_time_s:
            continue
        time_s = apply_time_modifiers(time_s, "invention", skill_levels)

        # Invention + manufacturing materials costs in buy market
        inv_mats: List[Dict[str, Any]] = []
        mfg_mats: List[Dict[str, Any]] = []
        ok = True
        for m in r.get("invention_materials") or []:
            tid = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if tid <= 0 or qty <= 0:
                continue
            st = stats_for(buy_market.key, tid)
            sellp = fuzz_price(st, "sell", args.price_mode)
            buyp = fuzz_price(st, "buy", args.price_mode)
            if sellp <= 0 or buyp <= 0:
                ok = False
                break
            inv_mats.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": sellp, "unit_m3": tvol(tid)})
        if not ok:
            continue
        for m in r.get("manufacturing_materials") or []:
            tid = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if tid <= 0 or qty <= 0:
                continue
            st = stats_for(buy_market.key, tid)
            sellp = fuzz_price(st, "sell", args.price_mode)
            buyp = fuzz_price(st, "buy", args.price_mode)
            if sellp <= 0 or buyp <= 0:
                ok = False
                break
            mfg_mats.append({"type_id": tid, "name": tname(tid), "qty": qty, "unit_price": sellp, "unit_m3": tvol(tid)})
        if not ok:
            continue

        inv_attempts = safe_float(r.get("attempts_per_success"), 1.0)
        if inv_attempts <= 0:
            inv_attempts = 1.0

        inv_cost = sum(m["qty"] * m["unit_price"] for m in inv_mats) * inv_attempts
        mfg_cost = sum(m["qty"] * m["unit_price"] for m in mfg_mats)
        inv_mats_scaled: List[Dict[str, Any]] = []
        for m in inv_mats:
            qty_run = safe_float(m.get("qty"), 0.0) * float(inv_attempts)
            inv_mats_scaled.append(
                {
                    "type_id": safe_int(m.get("type_id"), 0),
                    "name": m.get("name") or tname(safe_int(m.get("type_id"), 0)),
                    "qty": qty_run,
                    "unit_price": safe_float(m.get("unit_price"), 0.0),
                    "extended": qty_run * safe_float(m.get("unit_price"), 0.0),
                }
            )
        mats_total_cost = inv_cost + mfg_cost

        # Job install cost (approx) for invention activity; we use invention + manufacturing mats
        mats_for_job = [{"type_id": m["type_id"], "qty": m["qty"], "unit_price": m["unit_price"]} for m in mfg_mats]
        job_cost = job_install_cost_per_run(adjusted_prices, inv_ci, mats_for_job, out_qty, fee_model)

        # Evaluate across sell markets (instant / patient)
        per_market: Dict[str, Any] = {}
        best_instant_key = None
        best_patient_key = None
        best_instant_profitph = -1e99
        best_patient_profitph = -1e99

        # For invention rows, treat "materials" as merged list (for UI); unit prices differ by mode not modeled deeply here.
        merged_instant = inv_mats + mfg_mats
        merged_patient = merged_instant  # keep same for simplicity

        for sm in sell_markets:
            out_buy = price_for(sm.key, pt, "buy")
            out_sell = price_for(sm.key, pt, "sell")
            if out_buy <= 0 or out_sell <= 0:
                continue

            inst = compute_mode_metrics(out_buy, out_qty, merged_instant, time_s, fee_model, job_cost, blueprint_cost=None, blueprint_runs=None)
            pat = compute_mode_metrics(out_sell, out_qty, merged_patient, time_s, fee_model, job_cost, blueprint_cost=None, blueprint_runs=None)
            # Replace materials_cost with our amortized invention cost model (since compute_mode_metrics sums per-run)
            # We do this so profit reflects invention attempts.
            inst["materials_cost"] = mats_total_cost
            inst["cost"] = mats_total_cost + inst["job_cost"] + inst["fees"]
            inst["profit"] = inst["revenue"] - inst["cost"]
            inst["roi"] = inst["profit"] / inst["cost"] if inst["cost"] > 0 else 0.0
            inst["profit_per_hour"] = inst["profit"] / (time_s / 3600.0) if time_s > 0 else 0.0

            pat["materials_cost"] = mats_total_cost
            pat["cost"] = mats_total_cost + pat["job_cost"] + pat["fees"]
            pat["profit"] = pat["revenue"] - pat["cost"]
            pat["roi"] = pat["profit"] / pat["cost"] if pat["cost"] > 0 else 0.0
            pat["profit_per_hour"] = pat["profit"] / (time_s / 3600.0) if time_s > 0 else 0.0

            conf, conf_details = compute_confidence_from_market_stats(stats_for(sm.key, pt))
            per_market[sm.key] = {
                "market_name": sm.name,
                "confidence": conf,
                "confidence_details": conf_details,
                "instant": inst,
                "patient": pat,
            }

            if conf >= args.best_min_confidence and fuzz_order_count(stats_for(sm.key, pt), "buy") >= args.min_output_buy_orders:
                if inst["profit_per_hour"] > best_instant_profitph:
                    best_instant_profitph = inst["profit_per_hour"]
                    best_instant_key = sm.key
                if pat["profit_per_hour"] > best_patient_profitph:
                    best_patient_profitph = pat["profit_per_hour"]
                    best_patient_key = sm.key

        if not per_market:
            continue
        if best_instant_key is None:
            best_instant_key = max(per_market.keys(), key=lambda k: per_market[k]["instant"]["profit_per_hour"])
        if best_patient_key is None:
            best_patient_key = max(per_market.keys(), key=lambda k: per_market[k]["patient"]["profit_per_hour"])

        t2_rows.append(
            {
                "category": "t2",
                "product_type_id": pt,
                "product_name": tname(pt),
                "output_qty": out_qty,
                "time_s": time_s,
                "attempts_per_success": inv_attempts,
                "invention_cost": inv_cost,
                "invention": {
                    "attempts_per_success": inv_attempts,
                    "cost_per_run": inv_cost,
                    "materials": inv_mats_scaled,
                },
                "manufacturing_cost": mfg_cost,
                "alternatives": {
                    "instant": summarize_alternatives(per_market, "instant", top_k=3),
                    "patient": summarize_alternatives(per_market, "patient", top_k=3),
                },
                "best_market": {"instant": best_instant_key, "patient": best_patient_key},
                "best_market_name": {"instant": cfg.markets[best_instant_key].name if best_instant_key in cfg.markets else best_instant_key,
                                     "patient": cfg.markets[best_patient_key].name if best_patient_key in cfg.markets else best_patient_key},
                "instant": per_market[best_instant_key]["instant"],
                "patient": per_market[best_patient_key]["patient"],
                "confidence": per_market[best_instant_key]["confidence"],
                "depth": None,
                "ttl": None,
            }
        )

    # -----------------------------
    # Sort and keep top rows
    # -----------------------------

    def sort_key(row: Dict[str, Any]) -> float:
        return safe_float((row.get("instant") or {}).get("profit_per_hour"), 0.0)

    manufacturing_rows.sort(key=sort_key, reverse=True)
    reaction_rows.sort(key=sort_key, reverse=True)
    t2_rows.sort(key=sort_key, reverse=True)
    refining_rows.sort(key=lambda r: safe_float(r.get("profit_per_m3"), 0.0), reverse=True)

    manufacturing_rows = manufacturing_rows[: args.max_rows]
    reaction_rows = reaction_rows[: args.max_rows]
    t2_rows = t2_rows[: args.max_rows]
    refining_rows = refining_rows[: args.max_rows]

    # -----------------------------
    # Depth validation + TTL for top-N (instant mode only)
    # -----------------------------

    structure_cache: Dict[int, List[Dict[str, Any]]] = {}

    def add_depth_and_ttl(rows: List[Dict[str, Any]], kind: str) -> None:
        top = rows[: min(args.depth_top_n, len(rows))]
        for row in top:
            try:
                best_market_key = (row.get("best_market") or {}).get("instant")
                if not best_market_key:
                    continue
                out_market = cfg.markets.get(best_market_key) or buy_market
                pt = safe_int(row.get("product_type_id"), 0)
                out_qty = safe_float(row.get("output_qty"), 1.0)
                mats = (row.get("instant") or {}).get("materials") or []
                # convert mats to depth format {type_id, qty, unit_price}
                depth_mats = [{"type_id": safe_int(m.get("type_id"), 0), "qty": safe_float(m.get("qty"), 0.0), "unit_price": safe_float(m.get("unit_price"), 0.0)} for m in mats]
                d = validate_depth_for_recipe(
                    input_market=buy_market,
                    output_market=out_market,
                    product_type_id=pt,
                    output_qty=out_qty,
                    materials=depth_mats,
                    input_slippage=args.input_slippage,
                    output_slippage=args.output_slippage,
                    depth_safety=args.depth_safety,
                    max_materials=args.depth_max_materials,
                    access_token=access_token,
                    structure_cache=structure_cache,
                )

                # compute depth-based expected profit at recommended runs using *mode fees* approximations
                runs = safe_int(d.get("recommended_runs"), 0)
                if runs <= 0:
                    row["depth"] = {**d, "guaranteed": False}
                    continue

                # Depth uses only subset mats; scale using ratios vs displayed per-run
                inst = row.get("instant") or {}
                base_cost_per_run = safe_float(inst.get("cost"), 0.0)
                base_profit_per_run = safe_float(inst.get("profit"), 0.0)
                base_revenue_per_run = safe_float(inst.get("revenue"), 0.0)

                depth_inputs_val = safe_float(((d.get("expected") or {}).get("inputs_value_depth_mats")), 0.0)
                depth_output_val = safe_float(((d.get("expected") or {}).get("output_value")), 0.0)

                # Estimate per-run depth-adjusted revenue and cost:
                # - Replace revenue with depth output average (value / units) * out_qty
                out_units = safe_float(((d.get("expected") or {}).get("output_units")), 0.0)
                out_avg = depth_output_val / out_units if out_units > 0 else 0.0
                depth_revenue_per_run = out_avg * out_qty

                # - Replace only the subset of materials with depth inputs; keep rest from base (approx)
                # Compute base cost of depth mats subset from inst materials list
                base_depth_mats_cost = 0.0
                depth_mat_ids = {safe_int(m.get("type_id"), 0) for m in (d.get("input") or {}).get("materials") or []}
                for m in (inst.get("materials") or []):
                    if safe_int(m.get("type_id"), 0) in depth_mat_ids:
                        base_depth_mats_cost += safe_float(m.get("extended"), 0.0)
                base_other_cost = base_cost_per_run - base_depth_mats_cost
                depth_cost_per_run = base_other_cost + (depth_inputs_val / max(runs, 1))

                depth_profit_per_run = depth_revenue_per_run - depth_cost_per_run
                profit_total = depth_profit_per_run * runs

                guaranteed = bool((d.get("input") or {}).get("filled_ok")) and bool((d.get("output") or {}).get("filled_ok")) and profit_total > 0

                row["depth"] = {
                    **d,
                    "expected": {
                        **(d.get("expected") or {}),
                        "revenue_per_run": depth_revenue_per_run,
                        "cost_per_run": depth_cost_per_run,
                        "profit_per_run": depth_profit_per_run,
                        "profit_total": profit_total,
                    },
                    "guaranteed": guaranteed,
                    "sell_market": out_market.key,
                    "sell_market_name": out_market.name,
                }

                # TTL based on region history if available
                if args.confidence_history and out_market.region_id and pt:
                    h = fetch_market_history(out_market.region_id, pt)
                    adv = avg_daily_volume(h, days=7)
                    if adv > 0:
                        hours = (runs * out_qty) / adv * 24.0
                        row["ttl"] = {"avg_daily_volume": adv, "hours": hours, "bucket": ttl_bucket(hours)}
                    else:
                        row["ttl"] = {"avg_daily_volume": 0.0, "hours": 0.0, "bucket": "unknown"}
            except Exception:
                continue

    add_depth_and_ttl(manufacturing_rows, "manufacturing")
    add_depth_and_ttl(reaction_rows, "reactions")
    add_depth_and_ttl(t2_rows, "t2")
    # refining depth validation is intentionally omitted here

    
    # -----------------------------
    # Build a compact type_info map for UI volume calculations
    # -----------------------------
    used_type_ids: Set[int] = set()

    def _add_type_id(tid: Any) -> None:
        t = safe_int(tid, 0)
        if t > 0:
            used_type_ids.add(t)

    def _add_materials(mats: Any) -> None:
        for m in mats or []:
            _add_type_id(m.get("type_id"))

    for row in manufacturing_rows:
        _add_type_id(row.get("product_type_id"))
        _add_type_id(row.get("blueprint_type_id"))
        _add_materials((row.get("instant") or {}).get("materials"))
    for row in reaction_rows:
        _add_type_id(row.get("product_type_id"))
        _add_type_id(row.get("blueprint_type_id"))
        _add_materials((row.get("instant") or {}).get("materials"))
    for row in t2_rows:
        _add_type_id(row.get("product_type_id"))
        _add_materials((row.get("instant") or {}).get("materials"))
        _add_materials((row.get("invention") or {}).get("materials"))
    for row in refining_rows:
        _add_type_id(row.get("input_type_id"))
        for o in row.get("outputs") or []:
            _add_type_id(o.get("type_id"))

    type_info: Dict[str, Any] = {}
    for tid in sorted(used_type_ids):
        type_info[str(tid)] = {"name": tname(tid), "volume": tvol(tid)}


# -----------------------------
    # Write output
    # -----------------------------
    out_path = Path(args.out)
    ensure_parent(out_path)

    payload = {
        "generated_at": utc_now_iso(),
        "type_info": type_info,
        "market": {
            "buy_market": {"key": buy_market.key, "name": buy_market.name, "kind": buy_market.kind},
            "sell_markets": [{"key": m.key, "name": m.name, "kind": m.kind} for m in sell_markets],
            "price_mode": args.price_mode,
            "pricing_scope": args.pricing_scope,
        },
        "character": character or None,
        "assumptions": {
            "fees": {
                "sales_tax": args.sales_tax,
                "broker_fee": args.broker_fee,
                "facility_tax": args.facility_tax,
                "structure_job_bonus": args.structure_job_bonus,
            },
            "depth": {
                "top_n": args.depth_top_n,
                "input_slippage": args.input_slippage,
                "output_slippage": args.output_slippage,
                "depth_safety": args.depth_safety,
                "max_materials": args.depth_max_materials,
            },
        },
        "manufacturing": manufacturing_rows,
        "reactions": reaction_rows,
        "refining": refining_rows,
        "t2": t2_rows,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[update_rankings] Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
