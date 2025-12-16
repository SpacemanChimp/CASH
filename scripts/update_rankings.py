#!/usr/bin/env python3
"""
update_rankings.py

Reads recipe data (data/recipes.json.gz), fetches market prices, computes rankings,
and writes docs/data/rankings.json for GitHub Pages.

Core concepts
- "Instant" mode: buy inputs from sell orders; sell outputs to buy orders.
  Conservative and fast to execute.
- "Patient" mode: place buy orders for inputs; list sell orders for outputs.
  Higher theoretical margin, but slower + broker fees.

New "Emperor" upgrades included:
1) Recommended batch size (depth validation)
   - For top N candidates per category, fetch ESI region orderbooks for the
     output + top-cost materials and compute:
       max runs by input depth (slippage cap)
       max runs by output depth (slippage cap)
       recommended runs (safety factor)
       expected profit at that scale using the orderbook, not just top-of-book

2) Confidence score (0-100)
   - Penalizes: low order count, low order volume, wide spread, high volatility,
     spiky percentile vs min/max.

3) Proper-ish industry costs and taxes
   - Job install cost approximated using ESI adjusted prices and system cost index
     from /industry/systems/.
   - Market fees include sales tax and (optionally) broker fee for patient mode.

4) T2 / Invention pipeline tab
   - Uses invention mappings precomputed in recipes.json.gz (activityID=8).
   - Computes invention-amortized cost/run and time/run.

5) Refining made durable
   - Refining recipes come from invTypeMaterials for ore/ice/moon/compressed-like items.
   - Refining uses region prices by default to avoid station scope wiping out data.
   - Separate liquidity thresholds for refining.

Notes / limitations (important for making real ISK)
- ESI region orderbooks do NOT include structure-only markets (citadels) except ranged
  orders. That means depth validation can undercount Perimeter/other structure liquidity.
  Baseline pricing uses Fuzzwork aggregates, which may include more complete data.
- This tool does not include hauling, liquidity/turnover constraints beyond confidence,
  or structure/rig/skill ME/TE rounding. It is meant to get you to the right lane fast,
  then you execute like an industrialist.

"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


# -----------------------------
# Config defaults
# -----------------------------

AGGREGATECSV_URL = "https://market.fuzzwork.co.uk/aggregatecsv.csv.gz"

ESI_BASE = "https://esi.evetech.net/latest"
ESI_DATASOURCE = "tranquility"

DEFAULT_REGION_ID = 10000002  # The Forge
DEFAULT_STATION_ID = 60003760  # Jita IV - Moon 4 - CNAP (NPC station)

# ---- Multi-market sell (arbitrage) configuration ----
# These are *market evaluation locations* (not where you must build).
# Inputs are priced from the buy market (default: Jita/The Forge),
# while outputs are valued at the best sell market among the configured sell markets.
#
# Note:
# - Fuzzwork "aggregatecsv" and ESI region market orders cover NPC-station markets.
# - Player structure markets (common in null-sec) require authenticated ESI calls to
#   /markets/structures/{structure_id}/ and a character that can access that structure.
#
# You can edit data/markets.json to change these without touching code.
MARKETS_CONFIG_PATH = Path("data/markets.json")

DEFAULT_MARKETS: List[Dict[str, Any]] = [
    {
        "key": "jita",
        "name": "Jita 4-4 (The Forge)",
        "kind": "station",
        "region_id": 10000002,
        "station_id": 60003760,
    },
    {
        "key": "amarr",
        "name": "Amarr (Domain)",
        "kind": "station",
        "region_id": 10000043,
        "station_id": 60008494,
    },
    {
        "key": "hek",
        "name": "Hek (Metropolis)",
        "kind": "station",
        "region_id": 10000042,
        "station_id": 60005686,
    },
    {
        "key": "dodixie",
        "name": "Dodixie (Sinq Laison)",
        "kind": "station",
        "region_id": 10000032,
        "station_id": 60011866,
    },
    {
        "key": "rens",
        "name": "Rens (Heimatar)",
        "kind": "station",
        "region_id": 10000030,
        "station_id": 60004588,
    },
    # Null-sec hub placeholders (typically structure markets -> require structure_id + SSO token)
    {"key": "r-ag7w", "name": "R-AG7W (structure market)", "kind": "structure", "structure_id": None},
    {"key": "e8-432", "name": "E8-432 (structure market)", "kind": "structure", "structure_id": None},
    {"key": "mj-5f9", "name": "MJ-5F9 (structure market)", "kind": "structure", "structure_id": None},
    {"key": "r10-gn", "name": "R10-GN (structure market)", "kind": "structure", "structure_id": None},
]

DEFAULT_BUY_MARKET = "jita"
DEFAULT_SELL_MARKETS = ["jita", "amarr", "hek", "dodixie", "rens"]


def load_markets_config(path: Path = MARKETS_CONFIG_PATH) -> Dict[str, Any]:
    """Loads market config from JSON, falling back to defaults."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "buy_market": DEFAULT_BUY_MARKET,
        "sell_markets": DEFAULT_SELL_MARKETS,
        "markets": DEFAULT_MARKETS,
    }

# Jita solar system ID (useful default for showing indices, though most builders use other systems)
DEFAULT_SYSTEM_ID = 30000142  # Jita

RECIPES_PATH = Path("data/recipes.json.gz")
RANKINGS_PATH = Path("docs/data/rankings.json")

CACHE_DIR = Path(".cache")
AGG_CSV_CACHE = CACHE_DIR / "aggregatecsv.csv.gz"

# Orderbook validation limits
ESI_MAX_PAGES_PER_TYPE = 50
ESI_TIMEOUT = 60
ESI_SLEEP_S = 0.15  # be gentle


SSO_TOKEN_URL = "https://login.eveonline.com/v2/oauth/token"


def get_sso_access_token_from_env() -> Optional[str]:
    """
    Optional support for structure market orders (requires auth).

    Provide either:
      - EVE_SSO_ACCESS_TOKEN  (short-lived access token), or
      - EVE_SSO_CLIENT_ID + EVE_SSO_CLIENT_SECRET + EVE_SSO_REFRESH_TOKEN

    Required ESI scope for structure markets:
      esi-markets.structure_markets.v1
    """
    access = (os.getenv("EVE_SSO_ACCESS_TOKEN") or "").strip()
    if access:
        return access

    client_id = (os.getenv("EVE_SSO_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("EVE_SSO_CLIENT_SECRET") or "").strip()
    refresh = (os.getenv("EVE_SSO_REFRESH_TOKEN") or "").strip()
    if not (client_id and client_secret and refresh):
        return None

    try:
        resp = requests.post(
            SSO_TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh},
            auth=requests.auth.HTTPBasicAuth(client_id, client_secret),
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        tok = (resp.json().get("access_token") or "").strip()
        return tok or None
    except Exception as e:
        print(f"[update_rankings] Failed to refresh ESI token (structure markets will be skipped): {e}")
        return None


def fetch_structure_orders(structure_id: int, access_token: str) -> List[Dict[str, Any]]:
    """Fetch all market orders for a structure (requires esi-markets.structure_markets.v1)."""
    orders: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = f"{ESI_BASE}/markets/structures/{structure_id}/"
        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {access_token}",
        }
        r = requests.get(
            url,
            params={"datasource": "tranquility", "page": page},
            headers=headers,
            timeout=60,
        )
        if r.status_code == 403:
            raise RuntimeError("403 forbidden (missing access or missing esi-markets.structure_markets.v1 scope)")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            break
        orders.extend(data)
        x_pages = safe_int(r.headers.get("X-Pages"), page)
        if page >= x_pages:
            break
        page += 1
        time.sleep(ESI_SLEEP_S)
    return orders


def build_stats_from_orders(orders: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build a Fuzzwork-like stats dict from structure orders:
      stats[type_id]["buy"|"sell"] = {min,max,median,weightedAverage,percentile,volume,orderCount,stddev}
    """
    acc: Dict[int, Dict[str, List[Tuple[float, int]]]] = {}
    for o in orders:
        try:
            tid = int(o.get("type_id"))
            side = "buy" if bool(o.get("is_buy_order")) else "sell"
            price = float(o.get("price") or 0.0)
            vol = int(o.get("volume_remain") or o.get("volume_total") or 0)
        except Exception:
            continue
        if tid <= 0 or price <= 0 or vol <= 0:
            continue
        acc.setdefault(tid, {"buy": [], "sell": []})[side].append((price, vol))

    def weighted_quantile(pv: List[Tuple[float, int]], q: float) -> float:
        if not pv:
            return 0.0
        total = sum(v for _, v in pv)
        if total <= 0:
            return 0.0
        target = q * total
        cum = 0
        for p, v in sorted(pv, key=lambda x: x[0]):
            cum += v
            if cum >= target:
                return p
        return sorted(pv, key=lambda x: x[0])[-1][0]

    stats: Dict[int, Dict[str, Any]] = {}
    for tid, sides in acc.items():
        row: Dict[str, Any] = {}
        for side in ("buy", "sell"):
            pv = sides.get(side) or []
            if not pv:
                continue
            total_vol = sum(v for _, v in pv)
            wavg = sum(p * v for p, v in pv) / total_vol if total_vol > 0 else 0.0
            var = sum(v * (p - wavg) ** 2 for p, v in pv) / total_vol if total_vol > 0 else 0.0
            stddev = var ** 0.5
            prices = [p for p, _ in pv]

            # Approximate EVE "5% price" behavior:
            #  - sell side: low-end 5% (conservative buying from sells)
            #  - buy side: high-end 5% (conservative selling into buys) -> 95th percentile
            perc = weighted_quantile(pv, 0.05 if side == "sell" else 0.95)

            row[side] = {
                "min": min(prices),
                "max": max(prices),
                "median": weighted_quantile(pv, 0.5),
                "weightedAverage": wavg,
                "stddev": stddev,
                "volume": float(total_vol),
                "orderCount": len(pv),
                "percentile": perc,
            }
        if row:
            stats[tid] = row
    return stats


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1 else x)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


# -----------------------------
# HTTP helpers
# -----------------------------

def http_get_json(url: str, timeout: int = 60, retries: int = 3, sleep_s: float = 1.0) -> Any:
    headers = {
        "User-Agent": "eve-money-button/1.0 (+https://github.com/)",
        "Accept": "application/json",
    }
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(sleep_s * (1.5 ** i))
    raise last_err or RuntimeError("http_get_json failed")


def http_get(url: str, timeout: int = 60, retries: int = 3, sleep_s: float = 1.0, stream: bool = False) -> requests.Response:
    headers = {
        "User-Agent": "eve-money-button/1.0 (+https://github.com/)",
        "Accept": "*/*",
    }
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(sleep_s * (1.5 ** i))
    raise last_err or RuntimeError("http_get failed")


def download_to(url: str, dest: Path, timeout: int = 120) -> Dict[str, str]:
    ensure_parent(dest)
    with http_get(url, timeout=timeout, retries=3, stream=True) as r:
        headers = {k: v for k, v in r.headers.items()}
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return headers


def download_if_stale(url: str, dest: Path, max_age_s: int) -> Dict[str, str]:
    """
    Download dest if missing or older than max_age_s.
    Returns headers if downloaded, else {}.
    """
    if dest.exists():
        age = time.time() - dest.stat().st_mtime
        if age < max_age_s:
            return {}
    print(f"[update_rankings] Downloading: {url}")
    return download_to(url, dest)


# -----------------------------
# Recipes
# -----------------------------

def load_recipes(recipes_path: Path) -> Dict[str, Any]:
    with gzip.open(recipes_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def maybe_build_recipes(recipes_path: Path, force: bool = False) -> None:
    if recipes_path.exists() and not force:
        return
    print("[update_rankings] recipes.json.gz missing or forced; building from SDE (one-time-ish)…")
    import subprocess
    cmd = [sys.executable, "scripts/build_recipes.py", "--out", str(recipes_path)]
    if force:
        cmd.append("--force")
    subprocess.check_call(cmd)


# -----------------------------
# Market aggregates (Fuzzwork)
# -----------------------------

def load_region_prices_from_aggregatecsv(cache_path: Path, region_id: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, str]]:
    """
    Returns:
      prices[type_id] = {"buy": {...}, "sell": {...}}
      headers from download (if we downloaded), else {}.
    """
    headers = download_if_stale(AGGREGATECSV_URL, cache_path, max_age_s=25 * 60)

    prices: Dict[int, Dict[str, Any]] = {}
    with gzip.open(cache_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            what = row.get("what") or ""
            parts = what.split("|")
            if len(parts) != 3:
                continue
            try:
                rid = int(parts[0])
                if rid != region_id:
                    continue
                tid = int(parts[1])
                is_buy = parts[2].lower() == "true"
            except Exception:
                continue

            def fnum(k: str) -> float:
                v = row.get(k)
                try:
                    return float(v) if v not in (None, "") else 0.0
                except Exception:
                    return 0.0

            def inum(k: str) -> int:
                v = row.get(k)
                try:
                    return int(float(v)) if v not in (None, "") else 0
                except Exception:
                    return 0

            side = "buy" if is_buy else "sell"
            prices.setdefault(tid, {})
            prices[tid][side] = {
                "weightedAverage": fnum("weightedaverage"),
                "max": fnum("maxval"),
                "min": fnum("minval"),
                "stddev": fnum("stddev"),
                "median": fnum("median"),
                "volume": fnum("volume"),
                "orderCount": inum("numorders"),
                "percentile": fnum("fivepercent"),
            }

    return prices, headers




def load_multi_region_prices_from_aggregatecsv(
    cache_path: Path, region_ids: Set[int]
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], Dict[str, str]]:
    """
    Loads Fuzzwork region aggregates for multiple regions in a single pass.

    Returns:
      prices_by_region[region_id][type_id] = {"buy": {...}, "sell": {...}}
      headers (HTTP response headers from download if we downloaded)
    """
    headers: Dict[str, str] = {}
    if not cache_path.exists():
        print(f"[update_rankings] Downloading region aggregates CSV: {AGGREGATECSV_URL}")
        headers = download_to(AGGREGATECSV_URL, cache_path)

    # Initialize dicts so missing regions still exist
    prices_by_region: Dict[int, Dict[int, Dict[str, Any]]] = {int(rid): {} for rid in region_ids}

    with gzip.open(cache_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            what = row.get("what") or ""
            parts = what.split("|")
            if len(parts) != 3:
                continue
            try:
                rid = int(parts[0])
                if rid not in prices_by_region:
                    continue
                tid = int(parts[1])
                is_buy = parts[2].lower() == "true"
            except ValueError:
                continue

            def fnum(k: str) -> float:
                v = row.get(k)
                try:
                    return float(v) if v not in (None, "") else 0.0
                except ValueError:
                    return 0.0

            def inum(k: str) -> int:
                v = row.get(k)
                try:
                    return int(float(v)) if v not in (None, "") else 0
                except ValueError:
                    return 0

            side = "buy" if is_buy else "sell"
            stats = prices_by_region[rid].setdefault(tid, {})
            stats[side] = {
                "weightedAverage": fnum("weightedaverage"),
                "max": fnum("maxval"),
                "min": fnum("minval"),
                "stddev": fnum("stddev"),
                "median": fnum("median"),
                "volume": fnum("volume"),
                "orderCount": inum("numorders"),
                "percentile": fnum("fivepercent"),
            }

    return prices_by_region, headers
def fuzz_price(stats: Dict[str, Any], side: str, mode: str) -> float:
    """
    side: "buy" or "sell"
    mode:
      - "minmax": buy=max buy, sell=min sell (top-of-book)
      - "percentile": use 5% average (fivepercent) when available, else fallback
      - "weighted": weightedAverage
    """
    if not stats or side not in stats:
        return 0.0
    s = stats[side]
    if mode == "weighted":
        return float(s.get("weightedAverage") or 0.0)
    if mode == "percentile":
        p = float(s.get("percentile") or 0.0)
        if p > 0:
            return p
        # fallback
        if side == "sell":
            return float(s.get("min") or 0.0)
        return float(s.get("max") or 0.0)
    # minmax
    if side == "sell":
        return float(s.get("min") or 0.0)
    return float(s.get("max") or 0.0)


# -----------------------------
# ESI helpers
# -----------------------------

def esi_url(path: str, **params: Any) -> str:
    # Always include datasource.
    q = {"datasource": ESI_DATASOURCE}
    q.update({k: v for k, v in params.items() if v is not None})
    qs = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in q.items())
    if not path.startswith("/"):
        path = "/" + path
    return f"{ESI_BASE}{path}?{qs}"


def fetch_adjusted_prices() -> Dict[int, Dict[str, float]]:
    """
    /markets/prices/ => [{type_id, adjusted_price, average_price}, ...]
    """
    url = esi_url("/markets/prices/")
    data = http_get_json(url, timeout=ESI_TIMEOUT, retries=3, sleep_s=1.0)
    out: Dict[int, Dict[str, float]] = {}
    for row in data:
        try:
            tid = int(row.get("type_id"))
        except Exception:
            continue
        out[tid] = {
            "adjusted": safe_float(row.get("adjusted_price"), 0.0),
            "average": safe_float(row.get("average_price"), 0.0),
        }
    return out


def fetch_system_cost_indices() -> Dict[int, Dict[str, float]]:
    """
    /industry/systems/ => [{solar_system_id, cost_indices:[{activity,cost_index},...]},...]
    Returns mapping: system_id -> {activity_name: cost_index}
    """
    url = esi_url("/industry/systems/")
    data = http_get_json(url, timeout=ESI_TIMEOUT, retries=3, sleep_s=1.0)
    out: Dict[int, Dict[str, float]] = {}
    for row in data:
        sid = safe_int(row.get("solar_system_id"), 0)
        if sid <= 0:
            continue
        cmap: Dict[str, float] = {}
        for ci in row.get("cost_indices", []) or []:
            act = str(ci.get("activity") or "")
            val = safe_float(ci.get("cost_index"), 0.0)
            if act:
                cmap[act] = val
        out[sid] = cmap
    return out


def fetch_market_history(region_id: int, type_id: int) -> List[Dict[str, Any]]:
    """
    /markets/{region_id}/history/?type_id=...
    Returns list of daily rows.
    """
    url = esi_url(f"/markets/{region_id}/history/", type_id=type_id)
    return http_get_json(url, timeout=ESI_TIMEOUT, retries=3, sleep_s=1.0)


ORDERS_CACHE: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}


def fetch_region_orders(region_id: int, type_id: int, order_type: str, max_pages: int = ESI_MAX_PAGES_PER_TYPE) -> List[Dict[str, Any]]:
    """
    /markets/{region_id}/orders/?order_type=buy|sell&type_id=...&page=...
    """
    cache_key = (region_id, type_id, order_type)
    cached = ORDERS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    orders: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = esi_url(f"/markets/{region_id}/orders/", order_type=order_type, type_id=type_id, page=page)
        r = http_get(url, timeout=ESI_TIMEOUT, retries=3, sleep_s=1.0, stream=False)
        try:
            data = r.json()
        except Exception:
            data = []
        if isinstance(data, list):
            orders.extend(data)
        pages = safe_int(r.headers.get("X-Pages"), 1)
        if page >= pages:
            break
        page += 1
        if page > max_pages:
            break
        time.sleep(ESI_SLEEP_S)
    ORDERS_CACHE[cache_key] = orders
    return orders


# -----------------------------
# Confidence scoring
# -----------------------------

def compute_confidence_from_stats(stats: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Confidence score 0-100 computed from Fuzzwork aggregate stats.

    This is designed to:
    - punish 1-order markets and wide spreads
    - prefer thick books (orderCount + volume)
    - punish spiky books (percentile far from min/max)
    - punish high relative stddev
    """
    if not stats or "buy" not in stats or "sell" not in stats:
        return 0, {"reason": "missing-stats"}

    b = stats["buy"]
    s = stats["sell"]

    buy_max = safe_float(b.get("max"), 0.0)
    sell_min = safe_float(s.get("min"), 0.0)

    buy_orders = safe_int(b.get("orderCount"), 0)
    sell_orders = safe_int(s.get("orderCount"), 0)
    buy_vol = safe_float(b.get("volume"), 0.0)
    sell_vol = safe_float(s.get("volume"), 0.0)

    spread = 1.0
    if sell_min > 0 and buy_max > 0:
        spread = (sell_min - buy_max) / sell_min
        spread = max(0.0, min(spread, 1.0))

    # Spikiness: percentile vs min/max divergence.
    sell_pct = safe_float(s.get("percentile"), 0.0)
    buy_pct = safe_float(b.get("percentile"), 0.0)
    sell_spike = 0.0
    if sell_pct > 0 and sell_min > 0:
        sell_spike = max(0.0, (sell_pct - sell_min) / sell_pct)
    buy_spike = 0.0
    if buy_pct > 0 and buy_max > 0:
        buy_spike = max(0.0, (buy_max - buy_pct) / buy_pct)

    # Volatility proxy: stddev / weightedAverage
    buy_wa = safe_float(b.get("weightedAverage"), 0.0)
    sell_wa = safe_float(s.get("weightedAverage"), 0.0)
    buy_std = safe_float(b.get("stddev"), 0.0)
    sell_std = safe_float(s.get("stddev"), 0.0)
    rel_vol = 0.0
    parts = []
    if buy_wa > 0:
        parts.append(buy_std / buy_wa)
    if sell_wa > 0:
        parts.append(sell_std / sell_wa)
    if parts:
        rel_vol = sum(parts) / len(parts)

    # Normalize sub-scores.
    # These targets are deliberately "industrialist safe" rather than trader-precise.
    oc = (buy_orders + sell_orders) / 2.0
    oc_score = clamp01(math.log10(oc + 1) / math.log10(200 + 1))  # 200 avg orders saturates
    vol_score = clamp01(math.log10((buy_vol + sell_vol) / 2.0 + 1) / math.log10(1e7 + 1))  # saturate around 10M units

    spread_score = 1.0 - clamp01(spread / 0.08)  # 8% spread is already annoying for "press button"
    spike_score = 1.0 - clamp01((sell_spike + buy_spike) / 2.0 / 0.20)  # 20% divergence is suspicious
    volat_score = 1.0 - clamp01(rel_vol / 0.25)  # 25% rel stddev = unstable

    # Weighted blend.
    score01 = (
        0.28 * oc_score +
        0.22 * vol_score +
        0.20 * spread_score +
        0.15 * spike_score +
        0.15 * volat_score
    )
    score = int(round(100 * clamp01(score01)))

    details = {
        "orderCount_avg": oc,
        "bookVolume_avg": (buy_vol + sell_vol) / 2.0,
        "spread": spread,
        "sell_spike": sell_spike,
        "buy_spike": buy_spike,
        "rel_vol": rel_vol,
        "components": {
            "orderCount": oc_score,
            "bookVolume": vol_score,
            "spread": spread_score,
            "spike": spike_score,
            "volatility": volat_score,
        },
    }
    return score, details


def attach_history_liquidity(
    confidence_details: Dict[str, Any],
    history_rows: List[Dict[str, Any]],
    window_days: int = 14,
) -> None:
    """
    Adds history-based metrics to confidence_details if available.
    """
    if not history_rows:
        return
    # Last N days.
    rows = history_rows[-window_days:] if len(history_rows) >= window_days else history_rows[:]
    vols = [safe_float(r.get("volume"), 0.0) for r in rows]
    avgs = [safe_float(r.get("average"), 0.0) for r in rows]
    if not vols:
        return
    avg_daily_vol = sum(vols) / len(vols)
    # Price volatility from history (stddev / mean)
    mean_price = sum(avgs) / len(avgs) if avgs else 0.0
    if mean_price > 0 and avgs:
        var = sum((p - mean_price) ** 2 for p in avgs) / len(avgs)
        std = math.sqrt(var)
        hist_rel = std / mean_price
    else:
        hist_rel = 0.0

    confidence_details["history"] = {
        "window_days": len(rows),
        "avg_daily_volume": avg_daily_vol,
        "rel_volatility": hist_rel,
    }


def bump_confidence_with_history(base_score: int, details: Dict[str, Any]) -> int:
    """
    Adjust base_score using history metrics when present.
    """
    hist = details.get("history")
    if not hist:
        return base_score
    vol = safe_float(hist.get("avg_daily_volume"), 0.0)
    rel_vol = safe_float(hist.get("rel_volatility"), 0.0)

    # Volume bonus: saturate around 2000/day for big items, but still helps for modules.
    vol_bonus = 10.0 * clamp01(math.log10(vol + 1) / math.log10(20000 + 1))
    # Volatility penalty: >20% is rough.
    vol_penalty = 12.0 * clamp01(rel_vol / 0.20)

    s = base_score + vol_bonus - vol_penalty
    return int(round(max(0.0, min(100.0, s))))


# -----------------------------
# Depth validation (recommended runs)
# -----------------------------

@dataclass
class OrderBook:
    best_price: float
    orders: List[Tuple[float, int]]  # (price, volume_remain)


def normalize_orders(orders: List[Dict[str, Any]], side: str) -> OrderBook:
    """
    side: 'sell' => sort asc, 'buy' => sort desc
    """
    rows: List[Tuple[float, int]] = []
    for o in orders:
        try:
            price = float(o.get("price"))
            vol = int(o.get("volume_remain") or 0)
            if price <= 0 or vol <= 0:
                continue
            rows.append((price, vol))
        except Exception:
            continue
    if not rows:
        return OrderBook(best_price=0.0, orders=[])
    if side == "sell":
        rows.sort(key=lambda x: x[0])
    else:
        rows.sort(key=lambda x: -x[0])
    best = rows[0][0]
    return OrderBook(best_price=best, orders=rows)


def max_units_with_slippage(book: OrderBook, side: str, slippage: float) -> int:
    """
    side:
      - 'sell' means you're buying from sell orders => average <= best*(1+slippage)
      - 'buy' means you're selling into buy orders => average >= best*(1-slippage)
    """
    if not book.orders or book.best_price <= 0:
        return 0
    best = book.best_price
    if side == "sell":
        threshold = best * (1.0 + slippage)
        cmp_ok = lambda avg: avg <= threshold
    else:
        threshold = best * (1.0 - slippage)
        cmp_ok = lambda avg: avg >= threshold

    total_units = 0
    total_value = 0.0
    last_ok_units = 0

    for price, vol in book.orders:
        # take full order
        total_value += price * vol
        total_units += vol
        avg = total_value / total_units if total_units > 0 else 0.0
        if cmp_ok(avg):
            last_ok_units = total_units
        else:
            break

    return int(last_ok_units)


def take_units(book: OrderBook, qty: int) -> Tuple[int, float, float]:
    """
    Returns (filled_qty, total_value, avg_price)
    """
    if qty <= 0 or not book.orders:
        return 0, 0.0, 0.0
    remain = qty
    total = 0.0
    filled = 0
    for price, vol in book.orders:
        if remain <= 0:
            break
        take = vol if vol <= remain else remain
        total += price * take
        filled += take
        remain -= take
    avg = total / filled if filled > 0 else 0.0
    return filled, total, avg


def pick_depth_materials(materials: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    """
    Depth validation across *every* material can explode calls.
    We focus on the max_items inputs by per-run ISK share (extended).
    """
    if not materials:
        return []
    # Expect materials already have 'extended_instant' field.
    ms = [m for m in materials if safe_float(m.get("extended_instant"), 0.0) > 0]
    ms.sort(key=lambda m: safe_float(m.get("extended_instant"), 0.0), reverse=True)
    return ms[:max_items]


def validate_depth_for_recipe(
    input_region_id: int,
    output_region_id: int,
    output_type_id: int,
    output_qty_per_run: float,
    materials: List[Dict[str, Any]],
    input_slippage: float,
    output_slippage: float,
    safety: float,
    max_materials: int,
) -> Dict[str, Any]:
    """
    Compute recommended runs using ESI region orderbooks for output + top materials.
    """
    # Fetch output buy orders (instant sale) and sell orders (patient sale)
    out_buy_raw = fetch_region_orders(output_region_id, output_type_id, order_type="buy")
    time.sleep(ESI_SLEEP_S)
    out_sell_raw = fetch_region_orders(output_region_id, output_type_id, order_type="sell")

    out_buy = normalize_orders(out_buy_raw, side="buy")
    out_sell = normalize_orders(out_sell_raw, side="sell")

    # Inputs: focus on top-cost materials only (to keep ESI calls sane).
    depth_mats = pick_depth_materials(materials, max_items=max_materials)
    mat_books: Dict[int, OrderBook] = {}

    for m in depth_mats:
        tid = int(m["type_id"])
        raw = fetch_region_orders(input_region_id, tid, order_type="sell")
        time.sleep(ESI_SLEEP_S)
        mat_books[tid] = normalize_orders(raw, side="sell")

    # Runs by output depth (sell into buy orders).
    max_units_out = max_units_with_slippage(out_buy, side="buy", slippage=output_slippage)
    max_runs_output = int(max_units_out // max(1.0, output_qty_per_run))

    # Runs by input depth (buy inputs from sell orders).
    max_runs_input = 10**9
    limiting_input: Optional[str] = None
    for m in depth_mats:
        tid = int(m["type_id"])
        per_run = safe_float(m.get("qty"), 0.0)
        if per_run <= 0:
            continue
        book = mat_books.get(tid)
        if not book:
            continue
        max_units_in = max_units_with_slippage(book, side="sell", slippage=input_slippage)
        runs_here = int(max_units_in // per_run) if max_units_in > 0 else 0
        if runs_here < max_runs_input:
            max_runs_input = runs_here
            limiting_input = m.get("name")

    if max_runs_input == 10**9:
        max_runs_input = 0

    # Recommended runs
    cap = min(max_runs_input if max_runs_input > 0 else 10**9, max_runs_output if max_runs_output > 0 else 10**9)
    if cap == 10**9:
        cap = 0
    rec = int(math.floor(cap * safety)) if cap > 0 else 0
    if rec < 1:
        rec = 1

    # Compute expected profit at rec runs using orderbooks for output and selected inputs.
    # NOTE: For materials not depth-validated, we rely on precomputed extended cost in the ranking step.
    # Here we recompute only for depth_mats and output to reflect slippage.
    out_units = int(round(output_qty_per_run * rec))
    out_filled, out_value, out_avg = take_units(out_buy, out_units)
    out_fill_ratio = (out_filled / out_units) if out_units > 0 else 0.0

    input_value = 0.0
    input_fill_ratio = 1.0
    input_details: Dict[str, Any] = {}
    for m in depth_mats:
        tid = int(m["type_id"])
        per_run = safe_float(m.get("qty"), 0.0)
        units = int(round(per_run * rec))
        book = mat_books.get(tid)
        if not book:
            continue
        filled, cost, avg = take_units(book, units)
        input_value += cost
        if units > 0:
            input_fill_ratio = min(input_fill_ratio, filled / units)
        input_details[str(tid)] = {"requested": units, "filled": filled, "total": cost, "avg_price": avg}

    return {
        "recommended_runs": rec,
        "max_runs_input": max_runs_input if max_runs_input > 0 else None,
        "max_runs_output": max_runs_output if max_runs_output > 0 else None,
        "limiting_input": limiting_input,
        "output": {
            "best_buy": out_buy.best_price,
            "best_sell": out_sell.best_price,
            "requested": out_units,
            "filled": out_filled,
            "total": out_value,
            "avg_price": out_avg,
            "filled_ratio": out_fill_ratio,
        },
        "inputs": {
            "filled_ratio": input_fill_ratio,
            "details": input_details,
        },
        "slippage": {"input": input_slippage, "output": output_slippage, "safety": safety},
    }


# -----------------------------
# Cost model
# -----------------------------

@dataclass
class FeeModel:
    sales_tax: float        # applied to revenue
    broker_fee: float       # applied to order value when placing orders (patient)
    facility_tax: float     # applied to job cost
    structure_job_bonus: float  # e.g. 0.0 means none; 0.02 means 2% cheaper job cost


def job_install_cost(
    adjusted_prices: Dict[int, Dict[str, float]],
    mats: List[Dict[str, Any]],
    cost_index: float,
    fee_model: FeeModel,
) -> float:
    """
    Approximate job installation cost as:
      base_value = sum(adjusted_price(material) * qty)
      install = base_value * cost_index * (1 + facility_tax) * (1 - structure_job_bonus)
    """
    if cost_index <= 0:
        return 0.0
    base = 0.0
    for m in mats:
        tid = int(m["type_id"])
        qty = safe_float(m.get("qty"), 0.0)
        if qty <= 0:
            continue
        adj = safe_float(adjusted_prices.get(tid, {}).get("adjusted"), 0.0)
        if adj <= 0:
            # fallback to average price if adjusted missing
            adj = safe_float(adjusted_prices.get(tid, {}).get("average"), 0.0)
        if adj <= 0:
            continue
        base += adj * qty
    mult = (1.0 + fee_model.facility_tax) * (1.0 - fee_model.structure_job_bonus)
    return base * cost_index * mult


def compute_mode_metrics(
    output_qty: float,
    output_price: float,
    mats: List[Dict[str, Any]],
    mat_price_key: str,
    fee_model: FeeModel,
    include_broker_on_inputs: bool,
    include_broker_on_outputs: bool,
    include_sales_tax: bool,
    job_cost_per_run: float,
) -> Dict[str, Any]:
    """
    mats items must include mat_price_key and qty.
    """
    cost = 0.0
    mats_out: List[Dict[str, Any]] = []
    for m in mats:
        qty = safe_float(m.get("qty"), 0.0)
        p = safe_float(m.get(mat_price_key), 0.0)
        ext = qty * p
        cost += ext
        mats_out.append({**m, "unit_price_used": p, "extended_used": ext})

    revenue = output_qty * output_price
    fees = 0.0
    fee_breakdown = {}

    if include_sales_tax:
        st = revenue * fee_model.sales_tax
        fees += st
        fee_breakdown["sales_tax"] = st

    if include_broker_on_outputs:
        bf = revenue * fee_model.broker_fee
        fees += bf
        fee_breakdown["broker_sell"] = bf

    if include_broker_on_inputs and cost > 0:
        bf2 = cost * fee_model.broker_fee
        fees += bf2
        fee_breakdown["broker_buy"] = bf2

    if job_cost_per_run > 0:
        jc = job_cost_per_run
        fees += jc
        fee_breakdown["industry_install"] = jc

    profit = revenue - cost - fees
    roi = profit / cost if cost > 0 else None

    return {
        "cost": cost,
        "revenue": revenue,
        "fees": fees,
        "fee_breakdown": fee_breakdown,
        "profit": profit,
        "roi": roi,
        "materials": mats_out,
        "output_price": output_price,
    }


# -----------------------------
# Main ranking computation
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--region-id", type=int, default=DEFAULT_REGION_ID)
    ap.add_argument("--station-id", type=int, default=DEFAULT_STATION_ID)

    ap.add_argument("--markets-config", default=str(MARKETS_CONFIG_PATH),
                    help="Path to markets.json (buy market + sell market set).")
    ap.add_argument("--buy-market", default="",
                    help="Market key to price inputs from (overrides --region-id when found in markets config).")
    ap.add_argument("--sell-markets", default="",
                    help="Comma-separated market keys to consider for best sell market (defaults from markets.json).")


    ap.add_argument("--price-mode", choices=["minmax", "percentile", "weighted"], default="percentile",
                    help="How to select prices from aggregates. percentile is safest.")
    ap.add_argument("--pricing-scope", choices=["region", "station"], default="region",
                    help="Baseline pricing scope. Refining will still default to region to avoid blank outputs.")

    # Filtering
    ap.add_argument("--min-output-buy-orders", type=int, default=10)
    ap.add_argument("--min-blueprint-sell-orders", type=int, default=1)
    ap.add_argument("--min-job-time-s", type=int, default=60)
    ap.add_argument("--max-rows", type=int, default=500, help="Max rows per category in output (after sorting)")

    # Refining filtering
    ap.add_argument("--min-ref-input-sell-orders", type=int, default=5)
    ap.add_argument("--min-ref-output-buy-orders", type=int, default=3)

    # Confidence
    ap.add_argument("--confidence-history", action="store_true",
                    help="Augment confidence with ESI market history for top candidates (extra API calls).")

    # Depth validation
    ap.add_argument("--depth-top-n", type=int, default=25, help="Top N per category to validate with ESI orderbooks")
    ap.add_argument("--depth-max-materials", type=int, default=6, help="Max materials per recipe to depth-validate")
    ap.add_argument("--input-slippage", type=float, default=0.05)
    ap.add_argument("--output-slippage", type=float, default=0.05)
    ap.add_argument("--depth-safety", type=float, default=0.80)

    # Taxes / costs
    ap.add_argument("--sales-tax", type=float, default=0.03375, help="Sales tax rate (Accounting V ~3.375%%)")
    ap.add_argument("--broker-fee", type=float, default=0.01, help="Broker fee rate (varies by skills/standings)")
    ap.add_argument("--facility-tax", type=float, default=0.0, help="Structure facility tax applied to job install")
    ap.add_argument("--structure-job-bonus", type=float, default=0.0, help="Job cost reduction from rigs (e.g. 0.02)")
    ap.add_argument("--mfg-system-id", type=int, default=DEFAULT_SYSTEM_ID)
    ap.add_argument("--rx-system-id", type=int, default=DEFAULT_SYSTEM_ID)
    ap.add_argument("--inv-system-id", type=int, default=DEFAULT_SYSTEM_ID)

    ap.add_argument("--force-recipes", action="store_true", help="Rebuild recipes.json.gz")
    ap.add_argument("--out", default=str(RANKINGS_PATH))

    args = ap.parse_args(argv)
    # ---- Market configuration (buy market + sell markets) ----
    markets_cfg = load_markets_config(Path(args.markets_config))
    markets_list = markets_cfg.get("markets") or DEFAULT_MARKETS

    # Merge defaults so core hubs always exist
    market_by_key: Dict[str, Dict[str, Any]] = {}
    for mkt in DEFAULT_MARKETS:
        if isinstance(mkt, dict) and mkt.get("key"):
            market_by_key[str(mkt["key"])] = mkt
    for mkt in markets_list:
        if isinstance(mkt, dict) and mkt.get("key"):
            market_by_key[str(mkt["key"])] = mkt

    buy_market_key = (args.buy_market or markets_cfg.get("buy_market") or DEFAULT_BUY_MARKET).strip()
    if buy_market_key not in market_by_key:
        print(f"[update_rankings] Unknown buy market '{buy_market_key}', falling back to {DEFAULT_BUY_MARKET}")
        buy_market_key = DEFAULT_BUY_MARKET
    buy_market = market_by_key[buy_market_key]
    buy_region_id = safe_int(buy_market.get("region_id"), args.region_id)
    buy_station_id = safe_int(buy_market.get("station_id"), args.station_id)

    sell_markets_raw = (args.sell_markets or "").strip()
    if not sell_markets_raw:
        sell_markets_raw = ",".join(markets_cfg.get("sell_markets") or DEFAULT_SELL_MARKETS)
    sell_market_keys = [s.strip() for s in sell_markets_raw.split(",") if s.strip()]
    sell_market_keys = [k for k in sell_market_keys if k in market_by_key]
    if not sell_market_keys:
        sell_market_keys = DEFAULT_SELL_MARKETS[:]
    if buy_market_key not in sell_market_keys:
        sell_market_keys.insert(0, buy_market_key)

    # Back-compat vars (the rest of the script historically assumes a single region/station):
    region_id = buy_region_id
    station_id = buy_station_id

    maybe_build_recipes(RECIPES_PATH, force=args.force_recipes)
    recipes = load_recipes(RECIPES_PATH)

    types: Dict[int, Dict[str, Any]] = {int(k): v for k, v in (recipes.get("types") or {}).items()}

    # Load market aggregates (region scope baseline) for *all* regions we need:
    region_ids_needed: Set[int] = {region_id}
    for mk in sell_market_keys:
        rid = safe_int(market_by_key.get(mk, {}).get("region_id"), 0)
        if rid > 0:
            region_ids_needed.add(rid)

    region_stats_by_region, agg_headers = load_multi_region_prices_from_aggregatecsv(AGG_CSV_CACHE, region_ids_needed)
    region_stats = region_stats_by_region.get(region_id, {})

    # NOTE: station pricing-scope is still intentionally omitted in this bundle, because
    # station-level aggregates (Jita 4-4 only) will *miss* structure market activity.
    # Using region + robust percentile is the safer default.
    pricing_scope = args.pricing_scope
    stats = region_stats  # legacy alias (buy region)


    # Optional: load authenticated structure-market stats (null-sec hubs, private citadels, etc.)
    structure_stats_by_market: Dict[str, Dict[int, Dict[str, Any]]] = {}
    structure_market_keys = [
        k
        for k in sell_market_keys
        if (market_by_key.get(k) or {}).get("kind") == "structure"
        and safe_int((market_by_key.get(k) or {}).get("structure_id"), 0) > 0
    ]

    if structure_market_keys:
        token = get_sso_access_token_from_env()
        if not token:
            print("[update_rankings] Structure markets configured but no EVE SSO token provided; skipping structure markets.")
        else:
            for mk in structure_market_keys:
                sid = safe_int((market_by_key.get(mk) or {}).get("structure_id"), 0)
                if sid <= 0:
                    continue
                try:
                    print(f"[update_rankings] Fetching structure market orders for {mk} (structure_id={sid})")
                    orders = fetch_structure_orders(sid, token)
                    structure_stats_by_market[mk] = build_stats_from_orders(orders)
                except Exception as e:
                    print(f"[update_rankings] Failed to fetch structure market {mk}: {e}")

    def market_stats(mk: str, tid: int) -> Dict[str, Any]:
        mkt = market_by_key.get(mk) or {}
        kind = (mkt.get("kind") or "station").lower()
        if kind == "structure":
            return (structure_stats_by_market.get(mk) or {}).get(tid, {})
        rid = safe_int(mkt.get("region_id"), 0)
        if rid <= 0:
            return {}
        return (region_stats_by_region.get(rid) or {}).get(tid, {})

    def market_price(mk: str, tid: int, side: str) -> float:
        return fuzz_price(market_stats(mk, tid), side=side, mode=args.price_mode)

    def market_order_count(mk: str, tid: int, side: str) -> int:
        s = market_stats(mk, tid).get(side) or {}
        return safe_int(s.get("orderCount"), 0)


    # Load ESI adjusted prices and system cost indices (for job install)
    print("[update_rankings] Fetching ESI adjusted prices and cost indices…")
    adjusted_prices = fetch_adjusted_prices()
    sys_indices = fetch_system_cost_indices()

    mfg_cost_index = safe_float(sys_indices.get(args.mfg_system_id, {}).get("manufacturing"), 0.0)
    rx_cost_index = safe_float(sys_indices.get(args.rx_system_id, {}).get("reaction"), 0.0)
    inv_cost_index = safe_float(sys_indices.get(args.inv_system_id, {}).get("invention"), 0.0)

    fee_model = FeeModel(
        sales_tax=args.sales_tax,
        broker_fee=args.broker_fee,
        facility_tax=args.facility_tax,
        structure_job_bonus=args.structure_job_bonus,
    )

        # Helper: get stats for a type
    def tstats(tid: int, rid: Optional[int] = None) -> Dict[str, Any]:
        rr = region_stats_by_region.get(int(rid) if rid else region_id, {})
        return rr.get(tid, {})

    def price(tid: int, side: str, rid: Optional[int] = None) -> float:
        return fuzz_price(tstats(tid, rid=rid), side=side, mode=args.price_mode)

    def order_count(tid: int, side: str, rid: Optional[int] = None) -> int:
        s = tstats(tid, rid=rid).get(side) or {}
        return safe_int(s.get("orderCount"), 0)

    def best_sell_market(tid: int, side: str, min_orders: int) -> Tuple[Optional[str], float]:
        """Returns (market_key, price) for the best market among sell_market_keys."""
        best_key: Optional[str] = None
        best_p: float = 0.0
        for mk in sell_market_keys:
            rid = safe_int(market_by_key.get(mk, {}).get("region_id"), 0)
            if rid <= 0:
                continue
            if order_count(tid, side, rid=rid) < min_orders:
                continue
            p = price(tid, side, rid=rid)
            if p > best_p:
                best_p = p
                best_key = mk
        return best_key, best_p

# Compute manufacturing rows
    print("[update_rankings] Computing manufacturing rankings…")
    manufacturing_rows: List[Dict[str, Any]] = []
    for r in recipes.get("manufacturing") or []:
        bp_tid = safe_int(r.get("blueprint_type_id"), 0)
        prod_tid = safe_int(r.get("product_type_id"), 0)
        time_s = safe_int(r.get("time_s"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0)
        if bp_tid <= 0 or prod_tid <= 0:
            continue
        if time_s < args.min_job_time_s:
            continue

        # Blueprint must be purchasable (avoid limited editions / dead BPOs)
        bp_sell_orders = order_count(bp_tid, "sell")
        bp_cost = price(bp_tid, "sell")
        if bp_sell_orders < args.min_blueprint_sell_orders or bp_cost <= 0:
            continue

        # Output must be sellable in at least one of the configured sell markets.
        # Instant mode: sell to buy orders -> use the best buy price among sell_market_keys.
        out_buy_by_market: Dict[str, float] = {}
        out_sell_by_market: Dict[str, float] = {}
        out_buy_orders_by_market: Dict[str, int] = {}
        out_sell_orders_by_market: Dict[str, int] = {}

        best_buy_market: Optional[str] = None
        best_buy_price: float = 0.0
        best_sell_market_key: Optional[str] = None
        best_sell_price: float = 0.0

        for mk in sell_market_keys:
            pb = market_price(mk, prod_tid, "buy")
            ps = market_price(mk, prod_tid, "sell")
            ob = market_order_count(mk, prod_tid, "buy")
            os_ = market_order_count(mk, prod_tid, "sell")
            if pb <= 0 and ps <= 0 and ob <= 0 and os_ <= 0:
                continue
            out_buy_by_market[mk] = pb
            out_sell_by_market[mk] = ps
            out_buy_orders_by_market[mk] = ob
            out_sell_orders_by_market[mk] = os_
            if ob >= args.min_output_buy_orders and pb > best_buy_price:
                best_buy_price = pb
                best_buy_market = mk
            if os_ > 0 and ps > best_sell_price:
                best_sell_price = ps
                best_sell_market_key = mk

        if not best_buy_market or best_buy_price <= 0:
            continue

        if not best_sell_market_key or best_sell_price <= 0:
            best_sell_market_key = best_buy_market
            best_sell_price = out_sell_by_market.get(best_sell_market_key) or 0.0

        out_buy = best_buy_price
        out_sell = best_sell_price
        out_buy_market = best_buy_market
        out_sell_market = best_sell_market_key
        out_buy_region_id = safe_int(market_by_key.get(out_buy_market, {}).get("region_id"), 0)
        out_sell_region_id = safe_int(market_by_key.get(out_sell_market, {}).get("region_id"), 0)
        out_buy_structure_id = safe_int(market_by_key.get(out_buy_market, {}).get("structure_id"), 0)
        out_sell_structure_id = safe_int(market_by_key.get(out_sell_market, {}).get("structure_id"), 0)
        out_buy_market_kind = (market_by_key.get(out_buy_market, {}).get("kind") or "station")
        out_sell_market_kind = (market_by_key.get(out_sell_market, {}).get("kind") or "station")


        mats = r.get("materials") or []
        # Attach both sell/buy unit prices so the UI can toggle modes later
        mats2: List[Dict[str, Any]] = []
        missing = False
        for m in mats:
            mt = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if mt <= 0 or qty <= 0:
                continue
            sellp = price(mt, "sell")
            buyp = price(mt, "buy")
            if sellp <= 0 or buyp <= 0:
                # If either side missing, we can still compute instant mode if sell exists;
                # but patient mode will be incomplete. Keep if sell exists.
                if sellp <= 0:
                    missing = True
                    break
            mats2.append({
                "type_id": mt,
                "name": m.get("name") or types.get(mt, {}).get("name", f"type {mt}"),
                "qty": qty,
                "unit_price_sell": sellp if sellp > 0 else None,
                "unit_price_buy": buyp if buyp > 0 else None,
            })
        if missing or not mats2:
            continue

        # Precompute extended instant for depth material selection
        for m in mats2:
            m["extended_instant"] = safe_float(m.get("qty"), 0.0) * safe_float(m.get("unit_price_sell"), 0.0)


        job_cost = job_install_cost(adjusted_prices, mats2, mfg_cost_index, fee_model)

        instant = compute_mode_metrics(
            output_qty=out_qty,
            output_price=out_buy,
            mats=mats2,
            mat_price_key="unit_price_sell",
            fee_model=fee_model,
            include_broker_on_inputs=False,
            include_broker_on_outputs=False,
            include_sales_tax=True,
            job_cost_per_run=job_cost,
        )
        patient = compute_mode_metrics(
            output_qty=out_qty,
            output_price=out_sell,
            mats=mats2,
            mat_price_key="unit_price_buy",
            fee_model=fee_model,
            include_broker_on_inputs=True,
            include_broker_on_outputs=True,
            include_sales_tax=True,
            job_cost_per_run=job_cost,
        )

        # Confidence for the output market
        c0, c_details = compute_confidence_from_stats(market_stats(out_buy_market, prod_tid))
        confidence = c0
        if args.confidence_history:
            # We only fetch history later for top candidates (to avoid huge API spam)
            pass

        profit_per_hour = instant["profit"] / (time_s / 3600) if time_s > 0 else None

        payback_runs = None
        if bp_cost > 0 and instant["profit"] > 0:
            payback_runs = bp_cost / instant["profit"]

        manufacturing_rows.append({
            "category": "manufacturing",
            "blueprint_type_id": bp_tid,
            "blueprint_name": r.get("blueprint_name") or types.get(bp_tid, {}).get("name", f"type {bp_tid}"),
            "product_type_id": prod_tid,
            "product_name": r.get("product_name") or types.get(prod_tid, {}).get("name", f"type {prod_tid}"),
            "output_qty": out_qty,
            "time_s": time_s,
            "buy_market": buy_market_key,
            "sell_instant_market": out_buy_market,
            "sell_instant_market_name": (market_by_key.get(out_buy_market) or {}).get("name") or out_buy_market,
            "sell_instant_kind": out_buy_market_kind,
            "sell_instant_region_id": out_buy_region_id,
            "sell_instant_structure_id": out_buy_structure_id,
            "sell_patient_market": out_sell_market,
            "sell_patient_market_name": (market_by_key.get(out_sell_market) or {}).get("name") or out_sell_market,
            "sell_patient_kind": out_sell_market_kind,
            "sell_patient_region_id": out_sell_region_id,
            "sell_patient_structure_id": out_sell_structure_id,
            "sell_markets_considered": sell_market_keys,
            "output_buy_by_market": out_buy_by_market,
            "output_sell_by_market": out_sell_by_market,
            "output_buy_orders_by_market": out_buy_orders_by_market,
            "output_sell_orders_by_market": out_sell_orders_by_market,
            "blueprint_cost": bp_cost,
            "blueprint_sell_orders": bp_sell_orders,
            "confidence": confidence,
            "confidence_details": c_details,
            "instant": instant,
            "patient": patient,
            # convenience fields for existing UI
            "profit": instant["profit"],
            "profit_per_hour": profit_per_hour,
            "roi": instant["roi"],
            "cost": instant["cost"],
            "revenue": instant["revenue"],
            "fees": instant["fees"],
            "materials": instant["materials"],
            "payback_runs": payback_runs,
            "depth": None,
        })

    # Reactions
    print("[update_rankings] Computing reaction rankings…")
    reaction_rows: List[Dict[str, Any]] = []
    for r in recipes.get("reactions") or []:
        bp_tid = safe_int(r.get("blueprint_type_id"), 0)
        prod_tid = safe_int(r.get("product_type_id"), 0)
        time_s = safe_int(r.get("time_s"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0)
        if bp_tid <= 0 or prod_tid <= 0:
            continue
        if time_s < args.min_job_time_s:
            continue

        # Formula must be purchasable
        bp_sell_orders = order_count(bp_tid, "sell")
        bp_cost = price(bp_tid, "sell")
        if bp_sell_orders < args.min_blueprint_sell_orders or bp_cost <= 0:
            continue

        # Output must be sellable in at least one of the configured sell markets.
        out_buy_by_market: Dict[str, float] = {}
        out_sell_by_market: Dict[str, float] = {}
        out_buy_orders_by_market: Dict[str, int] = {}
        out_sell_orders_by_market: Dict[str, int] = {}

        best_buy_market: Optional[str] = None
        best_buy_price: float = 0.0
        best_sell_market_key: Optional[str] = None
        best_sell_price: float = 0.0

        for mk in sell_market_keys:
            pb = market_price(mk, prod_tid, "buy")
            ps = market_price(mk, prod_tid, "sell")
            ob = market_order_count(mk, prod_tid, "buy")
            os_ = market_order_count(mk, prod_tid, "sell")
            if pb <= 0 and ps <= 0 and ob <= 0 and os_ <= 0:
                continue
            out_buy_by_market[mk] = pb
            out_sell_by_market[mk] = ps
            out_buy_orders_by_market[mk] = ob
            out_sell_orders_by_market[mk] = os_
            if ob >= args.min_output_buy_orders and pb > best_buy_price:
                best_buy_price = pb
                best_buy_market = mk
            if os_ > 0 and ps > best_sell_price:
                best_sell_price = ps
                best_sell_market_key = mk

        if not best_buy_market or best_buy_price <= 0:
            continue

        if not best_sell_market_key or best_sell_price <= 0:
            best_sell_market_key = best_buy_market
            best_sell_price = out_sell_by_market.get(best_sell_market_key) or 0.0

        out_buy = best_buy_price
        out_sell = best_sell_price
        out_buy_market = best_buy_market
        out_sell_market = best_sell_market_key
        out_buy_region_id = safe_int(market_by_key.get(out_buy_market, {}).get("region_id"), 0)
        out_sell_region_id = safe_int(market_by_key.get(out_sell_market, {}).get("region_id"), 0)
        out_buy_structure_id = safe_int(market_by_key.get(out_buy_market, {}).get("structure_id"), 0)
        out_sell_structure_id = safe_int(market_by_key.get(out_sell_market, {}).get("structure_id"), 0)
        out_buy_market_kind = (market_by_key.get(out_buy_market, {}).get("kind") or "station")
        out_sell_market_kind = (market_by_key.get(out_sell_market, {}).get("kind") or "station")


        mats = r.get("materials") or []
        mats2: List[Dict[str, Any]] = []
        missing = False
        for m in mats:
            mt = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if mt <= 0 or qty <= 0:
                continue
            sellp = price(mt, "sell")
            buyp = price(mt, "buy")
            if sellp <= 0:
                missing = True
                break
            mats2.append({
                "type_id": mt,
                "name": m.get("name") or types.get(mt, {}).get("name", f"type {mt}"),
                "qty": qty,
                "unit_price_sell": sellp if sellp > 0 else None,
                "unit_price_buy": buyp if buyp > 0 else None,
            })
        if missing or not mats2:
            continue

        for m in mats2:
            m["extended_instant"] = safe_float(m.get("qty"), 0.0) * safe_float(m.get("unit_price_sell"), 0.0)


        job_cost = job_install_cost(adjusted_prices, mats2, rx_cost_index, fee_model)

        instant = compute_mode_metrics(
            output_qty=out_qty,
            output_price=out_buy,
            mats=mats2,
            mat_price_key="unit_price_sell",
            fee_model=fee_model,
            include_broker_on_inputs=False,
            include_broker_on_outputs=False,
            include_sales_tax=True,
            job_cost_per_run=job_cost,
        )
        patient = compute_mode_metrics(
            output_qty=out_qty,
            output_price=out_sell,
            mats=mats2,
            mat_price_key="unit_price_buy",
            fee_model=fee_model,
            include_broker_on_inputs=True,
            include_broker_on_outputs=True,
            include_sales_tax=True,
            job_cost_per_run=job_cost,
        )

        c0, c_details = compute_confidence_from_stats(market_stats(out_buy_market, prod_tid))
        confidence = c0

        profit_per_hour = instant["profit"] / (time_s / 3600) if time_s > 0 else None

        reaction_rows.append({
            "category": "reactions",
            "blueprint_type_id": bp_tid,
            "blueprint_name": r.get("blueprint_name") or types.get(bp_tid, {}).get("name", f"type {bp_tid}"),
            "product_type_id": prod_tid,
            "product_name": r.get("product_name") or types.get(prod_tid, {}).get("name", f"type {prod_tid}"),
            "output_qty": out_qty,
            "time_s": time_s,
            "buy_market": buy_market_key,
            "sell_instant_market": out_buy_market,
            "sell_instant_market_name": (market_by_key.get(out_buy_market) or {}).get("name") or out_buy_market,
            "sell_instant_kind": out_buy_market_kind,
            "sell_instant_region_id": out_buy_region_id,
            "sell_instant_structure_id": out_buy_structure_id,
            "sell_patient_market": out_sell_market,
            "sell_patient_market_name": (market_by_key.get(out_sell_market) or {}).get("name") or out_sell_market,
            "sell_patient_kind": out_sell_market_kind,
            "sell_patient_region_id": out_sell_region_id,
            "sell_patient_structure_id": out_sell_structure_id,
            "sell_markets_considered": sell_market_keys,
            "output_buy_by_market": out_buy_by_market,
            "output_sell_by_market": out_sell_by_market,
            "output_buy_orders_by_market": out_buy_orders_by_market,
            "output_sell_orders_by_market": out_sell_orders_by_market,
            "blueprint_cost": bp_cost,
            "blueprint_sell_orders": bp_sell_orders,
            "confidence": confidence,
            "confidence_details": c_details,
            "instant": instant,
            "patient": patient,
            "profit": instant["profit"],
            "profit_per_hour": profit_per_hour,
            "roi": instant["roi"],
            "cost": instant["cost"],
            "revenue": instant["revenue"],
            "fees": instant["fees"],
            "materials": instant["materials"],
            "payback_runs": None,
            "depth": None,
        })

    # Refining
    print("[update_rankings] Computing refining rankings…")
    refining_rows: List[Dict[str, Any]] = []
    for rr in recipes.get("refining") or []:
        in_tid = safe_int(rr.get("input_type_id"), 0)
        if in_tid <= 0:
            continue
        batch_units = safe_int(rr.get("batch_units"), 1)
        batch_m3 = safe_float(rr.get("batch_m3"), 0.0)
        if batch_units <= 0:
            batch_units = 1
        # Require ore/ice to have some sell depth
        if order_count(in_tid, "sell") < args.min_ref_input_sell_orders:
            continue

        in_cost_unit = price(in_tid, "sell")
        if in_cost_unit <= 0:
            continue
        in_cost = in_cost_unit * batch_units

        outs = rr.get("outputs") or []
        out_items: List[Dict[str, Any]] = []
        revenue = 0.0
        ok = False
        for o in outs:
            ot = safe_int(o.get("type_id"), 0)
            qty = safe_float(o.get("qty"), 0.0)
            if ot <= 0 or qty <= 0:
                continue
            # We assume you sell minerals to buy orders
            if order_count(ot, "buy") < args.min_ref_output_buy_orders:
                continue
            p = price(ot, "buy")
            if p <= 0:
                continue
            ext = qty * p
            out_items.append({
                "type_id": ot,
                "name": o.get("name") or types.get(ot, {}).get("name", f"type {ot}"),
                "qty": qty,
                "unit_price": p,
                "extended": ext,
            })
            revenue += ext
            ok = True

        if not ok or revenue <= 0:
            continue

        # Refining doesn't pay industry job install; taxes depend on structure. We'll leave job cost 0.
        fees = revenue * fee_model.sales_tax
        profit = revenue - in_cost - fees
        roi = profit / in_cost if in_cost > 0 else None
        profit_per_m3 = profit / batch_m3 if batch_m3 > 0 else None

        c0, c_details = compute_confidence_from_stats(tstats(in_tid))
        confidence = c0  # confidence of the input market (ore/ice)

        refining_rows.append({
            "category": "refining",
            "input_type_id": in_tid,
            "input_name": rr.get("input_name") or types.get(in_tid, {}).get("name", f"type {in_tid}"),
            "batch_units": batch_units,
            "batch_m3": batch_m3,
            "materials": out_items,  # show outputs in breakdown list
            "cost": in_cost,
            "revenue": revenue,
            "fees": fees,
            "profit": profit,
            "profit_per_m3": profit_per_m3,
            "roi": roi,
            "confidence": confidence,
            "confidence_details": c_details,
            "depth": None,
            "meta": rr.get("meta"),
        })

    # T2 / invention
    print("[update_rankings] Computing T2 invention rankings…")
    invention_map: Dict[int, Dict[str, Any]] = {}
    for inv in recipes.get("invention") or []:
        t2_bp = safe_int(inv.get("t2_blueprint_type_id"), 0)
        if t2_bp > 0:
            invention_map[t2_bp] = inv

    t2_rows: List[Dict[str, Any]] = []
    # Find manufacturing recipes whose blueprint is inventable.
    for r in recipes.get("manufacturing") or []:
        t2_bp = safe_int(r.get("blueprint_type_id"), 0)
        prod_tid = safe_int(r.get("product_type_id"), 0)
        time_s = safe_int(r.get("time_s"), 0)
        out_qty = safe_float(r.get("output_qty"), 1.0)
        if t2_bp <= 0 or prod_tid <= 0:
            continue
        if t2_bp not in invention_map:
            continue
        if time_s < args.min_job_time_s:
            continue
        # Output must be sellable in at least one of the configured sell markets.
        out_buy_by_market: Dict[str, float] = {}
        out_sell_by_market: Dict[str, float] = {}
        out_buy_orders_by_market: Dict[str, int] = {}
        out_sell_orders_by_market: Dict[str, int] = {}

        best_buy_market: Optional[str] = None
        best_buy_price: float = 0.0
        best_sell_market_key: Optional[str] = None
        best_sell_price: float = 0.0

        for mk in sell_market_keys:
            pb = market_price(mk, prod_tid, "buy")
            ps = market_price(mk, prod_tid, "sell")
            ob = market_order_count(mk, prod_tid, "buy")
            os_ = market_order_count(mk, prod_tid, "sell")
            if pb <= 0 and ps <= 0 and ob <= 0 and os_ <= 0:
                continue
            out_buy_by_market[mk] = pb
            out_sell_by_market[mk] = ps
            out_buy_orders_by_market[mk] = ob
            out_sell_orders_by_market[mk] = os_
            if ob >= args.min_output_buy_orders and pb > best_buy_price:
                best_buy_price = pb
                best_buy_market = mk
            if os_ > 0 and ps > best_sell_price:
                best_sell_price = ps
                best_sell_market_key = mk

        if not best_buy_market or best_buy_price <= 0:
            continue

        if not best_sell_market_key or best_sell_price <= 0:
            best_sell_market_key = best_buy_market
            best_sell_price = out_sell_by_market.get(best_sell_market_key) or 0.0

        out_buy = best_buy_price
        out_sell = best_sell_price
        out_buy_market = best_buy_market
        out_sell_market = best_sell_market_key
        out_buy_region_id = safe_int(market_by_key.get(out_buy_market, {}).get("region_id"), 0)
        out_sell_region_id = safe_int(market_by_key.get(out_sell_market, {}).get("region_id"), 0)
        out_buy_structure_id = safe_int(market_by_key.get(out_buy_market, {}).get("structure_id"), 0)
        out_sell_structure_id = safe_int(market_by_key.get(out_sell_market, {}).get("structure_id"), 0)
        out_buy_market_kind = (market_by_key.get(out_buy_market, {}).get("kind") or "station")
        out_sell_market_kind = (market_by_key.get(out_sell_market, {}).get("kind") or "station")


        inv = invention_map[t2_bp]
        p = safe_float(inv.get("probability"), 0.4)
        if p <= 0:
            p = 0.4
        runs_per_success = safe_int(inv.get("runs_per_success"), 1)
        if runs_per_success <= 0:
            runs_per_success = 1
        inv_time_s = safe_int(inv.get("time_s"), 0)

        # T2 manufacturing materials
        mats_mfg = r.get("materials") or []
        mats2: List[Dict[str, Any]] = []
        missing = False
        for m in mats_mfg:
            mt = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if mt <= 0 or qty <= 0:
                continue
            sellp = price(mt, "sell")
            buyp = price(mt, "buy")
            if sellp <= 0:
                missing = True
                break
            mats2.append({
                "type_id": mt,
                "name": m.get("name") or types.get(mt, {}).get("name", f"type {mt}"),
                "qty": qty,
                "unit_price_sell": sellp if sellp > 0 else None,
                "unit_price_buy": buyp if buyp > 0 else None,
            })
        if missing or not mats2:
            continue

        for m in mats2:
            m["extended_instant"] = safe_float(m.get("qty"), 0.0) * safe_float(m.get("unit_price_sell"), 0.0)

        # Invention materials per attempt
        inv_mats = inv.get("materials") or []
        inv_mats2: List[Dict[str, Any]] = []
        inv_missing = False
        for m in inv_mats:
            mt = safe_int(m.get("type_id"), 0)
            qty = safe_float(m.get("qty"), 0.0)
            if mt <= 0 or qty <= 0:
                continue
            sellp = price(mt, "sell")
            buyp = price(mt, "buy")
            if sellp <= 0:
                inv_missing = True
                break
            inv_mats2.append({
                "type_id": mt,
                "name": m.get("name") or types.get(mt, {}).get("name", f"type {mt}"),
                "qty": qty,
                "unit_price_sell": sellp if sellp > 0 else None,
                "unit_price_buy": buyp if buyp > 0 else None,
            })
        if inv_missing or not inv_mats2:
            continue

        # Job costs
        job_cost_mfg = job_install_cost(adjusted_prices, mats2, mfg_cost_index, fee_model)
        job_cost_inv = job_install_cost(adjusted_prices, inv_mats2, inv_cost_index, fee_model)

        # Compute invention amortized cost per *manufacturing run* in instant mode
        inv_attempt_cost = sum(safe_float(m["qty"]) * safe_float(m["unit_price_sell"]) for m in inv_mats2)
        inv_attempt_fees = job_cost_inv  # job install cost per attempt (no sales/broker here)
        expected_attempts_per_success = 1.0 / p if p > 0 else 10.0
        inv_cost_per_success = (inv_attempt_cost + inv_attempt_fees) * expected_attempts_per_success
        inv_cost_per_run = inv_cost_per_success / runs_per_success

        inv_time_per_success = inv_time_s * expected_attempts_per_success
        inv_time_per_run = inv_time_per_success / runs_per_success


        # Instant mode: buy inputs from sell, sell output to buy, include sales tax, include mfg job install, plus invention overhead
        instant_mfg = compute_mode_metrics(
            output_qty=out_qty,
            output_price=out_buy,
            mats=mats2,
            mat_price_key="unit_price_sell",
            fee_model=fee_model,
            include_broker_on_inputs=False,
            include_broker_on_outputs=False,
            include_sales_tax=True,
            job_cost_per_run=job_cost_mfg,
        )
        # Add invention overhead to cost/fees by treating it as extra cost
        instant_profit = instant_mfg["profit"] - inv_cost_per_run
        instant_cost_total = instant_mfg["cost"] + inv_cost_per_run
        instant_roi = instant_profit / instant_cost_total if instant_cost_total > 0 else None

        # Profit/hour considering manufacturing time only and combined time
        profit_per_mfg_hour = instant_profit / (time_s / 3600) if time_s > 0 else None
        total_time_s_per_run = time_s + inv_time_per_run
        profit_per_total_hour = instant_profit / (total_time_s_per_run / 3600) if total_time_s_per_run > 0 else None

        c0, c_details = compute_confidence_from_stats(market_stats(out_buy_market, prod_tid))
        confidence = c0

        t2_rows.append({
            "category": "t2",
            "product_type_id": prod_tid,
            "product_name": r.get("product_name") or types.get(prod_tid, {}).get("name", f"type {prod_tid}"),
            "t2_blueprint_type_id": t2_bp,
            "t2_blueprint_name": r.get("blueprint_name") or types.get(t2_bp, {}).get("name", f"type {t2_bp}"),
            "output_qty": out_qty,
            "time_s": time_s,
            "buy_market": buy_market_key,
            "sell_instant_market": out_buy_market,
            "sell_instant_market_name": (market_by_key.get(out_buy_market) or {}).get("name") or out_buy_market,
            "sell_instant_kind": out_buy_market_kind,
            "sell_instant_region_id": out_buy_region_id,
            "sell_instant_structure_id": out_buy_structure_id,
            "sell_patient_market": out_sell_market,
            "sell_patient_market_name": (market_by_key.get(out_sell_market) or {}).get("name") or out_sell_market,
            "sell_patient_kind": out_sell_market_kind,
            "sell_patient_region_id": out_sell_region_id,
            "sell_patient_structure_id": out_sell_structure_id,
            "sell_markets_considered": sell_market_keys,
            "output_buy_by_market": out_buy_by_market,
            "output_sell_by_market": out_sell_by_market,
            "output_buy_orders_by_market": out_buy_orders_by_market,
            "output_sell_orders_by_market": out_sell_orders_by_market,
            "confidence": confidence,
            "confidence_details": c_details,
            "materials": instant_mfg["materials"],
            "instant": {
                **instant_mfg,
                "profit": instant_profit,
                "roi": instant_roi,
                "cost_total": instant_cost_total,
                "profit_per_mfg_hour": profit_per_mfg_hour,
                "profit_per_total_hour": profit_per_total_hour,
                "time_total_s": total_time_s_per_run,
            },
            "invention": {
                "t1_blueprint_type_id": safe_int(inv.get("t1_blueprint_type_id"), 0),
                "t1_blueprint_name": inv.get("t1_blueprint_name"),
                "probability": p,
                "runs_per_success": runs_per_success,
                "time_s": inv_time_s,
                "expected_attempts_per_success": expected_attempts_per_success,
                "materials": inv_mats2,
                "attempt_cost": inv_attempt_cost,
                "job_cost_per_attempt": job_cost_inv,
                "cost_per_success": inv_cost_per_success,
                "cost_per_run": inv_cost_per_run,
                "time_per_run_s": inv_time_per_run,
            },
            "profit": instant_profit,
            "profit_per_hour": profit_per_total_hour,
            "roi": instant_roi,
            "cost": instant_cost_total,
            "revenue": instant_mfg["revenue"],
            "fees": instant_mfg["fees"],  # market+industry fees (not invention overhead)
            "depth": None,
        })

    # Sort and cap output
    def sort_and_cap(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        rows2 = [r for r in rows if r.get(key) is not None]
        rows2.sort(key=lambda r: safe_float(r.get(key), -1e18), reverse=True)
        return rows2[: args.max_rows]

    manufacturing_rows = sort_and_cap(manufacturing_rows, "profit_per_hour")
    reaction_rows = sort_and_cap(reaction_rows, "profit_per_hour")
    refining_rows = sort_and_cap(refining_rows, "profit_per_m3")
    t2_rows = sort_and_cap(t2_rows, "profit_per_hour")

    # Optional: history-based confidence bump for top candidates across all categories
    if args.confidence_history:
        print("[update_rankings] Fetching ESI market history for confidence bump (top candidates)…")
        # Gather unique output type ids among top candidates
        cand_type_ids: Set[int] = set()
        for r in manufacturing_rows[: args.depth_top_n]:
            cand_type_ids.add(int(r["product_type_id"]))
        for r in reaction_rows[: args.depth_top_n]:
            cand_type_ids.add(int(r["product_type_id"]))
        for r in t2_rows[: args.depth_top_n]:
            cand_type_ids.add(int(r["product_type_id"]))
        # Refining: use input ore type ids
        for r in refining_rows[: args.depth_top_n]:
            cand_type_ids.add(int(r["input_type_id"]))

        history_cache: Dict[int, List[Dict[str, Any]]] = {}
        for tid in sorted(cand_type_ids):
            try:
                hist = fetch_market_history(region_id, tid)
                history_cache[tid] = hist
                time.sleep(ESI_SLEEP_S)
            except Exception:
                continue

        def bump_row(row: Dict[str, Any], tid: int) -> None:
            details = row.get("confidence_details") or {}
            attach_history_liquidity(details, history_cache.get(tid) or [])
            row["confidence_details"] = details
            row["confidence"] = bump_confidence_with_history(int(row.get("confidence") or 0), details)

        for row in manufacturing_rows:
            bump_row(row, int(row["product_type_id"]))
        for row in reaction_rows:
            bump_row(row, int(row["product_type_id"]))
        for row in t2_rows:
            bump_row(row, int(row["product_type_id"]))
        for row in refining_rows:
            bump_row(row, int(row["input_type_id"]))

    # Depth validation for top N per category
    print("[update_rankings] Depth-validating top candidates with ESI orderbooks…")
    def do_depth(rows: List[Dict[str, Any]], is_refining: bool = False) -> None:
        top = rows[: args.depth_top_n]
        for row in top:
            try:
                if is_refining:
                    # Refining: treat batch as 1 "run" of batch_units input.
                    in_tid = int(row["input_type_id"])
                    batch_units = int(row["batch_units"])
                    # We'll validate only input depth + output minerals by value (from materials list)
                    # Convert "materials" (outputs) into a fake materials list for output depth isn't meaningful,
                    # so we treat "output depth" as the *minimum* mineral depth among top 3 minerals by value.
                    # For simplicity, we re-use validate_depth_for_recipe on the input market only.
                    depth = None
                    # Input depth: can we buy X batches before ore price rises?
                    # Build a pseudo recipe where output is the ore itself (so output depth is skipped).
                    # We'll skip output depth for refining; it's rarely the bottleneck vs ore acquisition.
                    raw = fetch_region_orders(region_id, in_tid, order_type="sell")
                    time.sleep(ESI_SLEEP_S)
                    book = normalize_orders(raw, side="sell")
                    max_units = max_units_with_slippage(book, side="sell", slippage=args.input_slippage)
                    max_batches = int(max_units // max(1, batch_units))
                    rec_batches = int(math.floor(max_batches * args.depth_safety)) if max_batches > 0 else 1
                    # Expected profit at rec_batches using ore orderbook avg
                    req_units = batch_units * max(1, rec_batches)
                    filled, ore_total, ore_avg = take_units(book, req_units)
                    revenue_total = safe_float(row.get("revenue"), 0.0) * max(1, rec_batches)
                    fees_total = revenue_total * args.sales_tax
                    profit_total = revenue_total - ore_total - fees_total
                    roi = profit_total / ore_total if ore_total > 0 else None

                    row["depth"] = {
                        "recommended_batches": max(1, rec_batches),
                        "max_batches_input": max_batches if max_batches > 0 else None,
                        "slippage": {"input": args.input_slippage, "safety": args.depth_safety},
                        "ore": {
                            "requested_units": req_units,
                            "filled_units": filled,
                            "total_cost": ore_total,
                            "avg_price": ore_avg,
                        },
                        "expected": {
                            "batches": max(1, rec_batches),
                            "revenue_total": revenue_total,
                            "cost_total": ore_total,
                            "fees_total": fees_total,
                            "profit_total": profit_total,
                            "roi": roi,
                        },
                        "note": "Refining depth uses ore sell orders only (minerals are typically very liquid).",
                    }
                else:
                    out_tid = int(row["product_type_id"])
                    out_qty = safe_float(row.get("output_qty"), 1.0)
                    mats = row.get("materials") or []
                    out_rid = safe_int(row.get("sell_instant_region_id"), 0)
                    if out_rid > 0:
                        depth = validate_depth_for_recipe(
                            input_region_id=region_id,
                            output_region_id=out_rid,
                            output_type_id=out_tid,
                            output_qty_per_run=out_qty,
                            materials=mats,
                            input_slippage=args.input_slippage,
                            output_slippage=args.output_slippage,
                            safety=args.depth_safety,
                            max_materials=args.depth_max_materials,
                        )
                    else:
                        depth = {}
                    # Attach scaled expectation at the recommended run count
                    rec = int(depth.get("recommended_runs") or 1)
                    inst = row.get("instant") or {}
                    mats_used = row.get("materials") or []

                    # Baseline totals at top-of-book (instant)
                    base_mats_total = sum(safe_float(m.get("extended_used"), 0.0) for m in mats_used) * rec
                    inv_over = safe_float((row.get("invention") or {}).get("cost_per_run"), 0.0)
                    base_inv_total = inv_over * rec

                    # Adjust a subset of inputs using orderbook average prices (slippage)
                    adj_mats_total = base_mats_total
                    input_details = (depth.get("inputs") or {}).get("details") or {}
                    for m in mats_used:
                        tid = str(int(m.get("type_id") or 0))
                        det = input_details.get(tid)
                        if not det:
                            continue
                        avg_p = safe_float(det.get("avg_price"), 0.0)
                        base_p = safe_float(m.get("unit_price_used"), 0.0)
                        qty_per_run = safe_float(m.get("qty"), 0.0)
                        if avg_p > 0 and base_p > 0 and qty_per_run > 0:
                            adj_mats_total += (avg_p - base_p) * qty_per_run * rec

                    adj_cost_total = adj_mats_total + base_inv_total

                    out_total = safe_float((depth.get("output") or {}).get("total"), 0.0)

                    # Recompute fees: keep non-sales-tax fees linear; recompute sales tax from scaled revenue.
                    fee_break = inst.get("fee_breakdown") or {}
                    base_sales_tax_per_run = safe_float(fee_break.get("sales_tax"), 0.0)
                    base_fees_per_run = safe_float(inst.get("fees"), 0.0)
                    other_fees_per_run = max(0.0, base_fees_per_run - base_sales_tax_per_run)
                    scaled_fees_total = other_fees_per_run * rec + out_total * args.sales_tax

                    expected_profit_total = out_total - adj_cost_total - scaled_fees_total
                    expected_profit_per_run = expected_profit_total / rec if rec > 0 else None
                    expected_roi = expected_profit_total / adj_cost_total if adj_cost_total > 0 else None

                    depth["expected"] = {
                        "runs": rec,
                        "revenue_total": out_total,
                        "cost_total": adj_cost_total,
                        "fees_total": scaled_fees_total,
                        "profit_total": expected_profit_total,
                        "profit_per_run": expected_profit_per_run,
                        "roi": expected_roi,
                    }

                    row["depth"] = depth
            except Exception as e:
                row["depth"] = {"error": str(e)}

    do_depth(manufacturing_rows, is_refining=False)
    do_depth(reaction_rows, is_refining=False)
    do_depth(t2_rows, is_refining=False)
    do_depth(refining_rows, is_refining=True)


    # Build compact type info for plan builder (volumes + names)
    used_type_ids: Set[int] = set()
    def note_type(tid: int) -> None:
        if tid and tid > 0:
            used_type_ids.add(int(tid))

    for row in manufacturing_rows:
        note_type(int(row.get("product_type_id") or 0))
        note_type(int(row.get("blueprint_type_id") or 0))
        for m in row.get("materials") or []:
            note_type(int(m.get("type_id") or 0))

    for row in reaction_rows:
        note_type(int(row.get("product_type_id") or 0))
        note_type(int(row.get("blueprint_type_id") or 0))
        for m in row.get("materials") or []:
            note_type(int(m.get("type_id") or 0))

    for row in t2_rows:
        note_type(int(row.get("product_type_id") or 0))
        note_type(int(row.get("t2_blueprint_type_id") or 0))
        inv = row.get("invention") or {}
        note_type(int(inv.get("t1_blueprint_type_id") or 0))
        for m in row.get("materials") or []:
            note_type(int(m.get("type_id") or 0))
        for m in (inv.get("materials") or []):
            note_type(int(m.get("type_id") or 0))

    for row in refining_rows:
        note_type(int(row.get("input_type_id") or 0))
        for o in row.get("materials") or []:
            note_type(int(o.get("type_id") or 0))

    type_info: Dict[str, Any] = {}
    for tid in sorted(used_type_ids):
        meta = types.get(tid) or {}
        type_info[str(tid)] = {
            "name": meta.get("name", f"type {tid}"),
            "volume": safe_float(meta.get("volume"), 0.0),
        }

    payload = {
        "generated_at": utc_now_iso(),
        "type_info": type_info,
        "market": {
            "region_id": region_id,
            "station_id": station_id,
            "buy_market": buy_market_key,
            "sell_markets": sell_market_keys,
            "markets": [market_by_key[k] for k in sorted(market_by_key.keys())],
            "pricing_scope": pricing_scope,
            "price_mode": args.price_mode,
        },
        "assumptions": {
            "modes": {
                "instant": "buy inputs from sell orders; sell output to buy orders",
                "patient": "place buy orders for inputs; list sell orders for output",
            },
            "taxes": {
                "sales_tax": args.sales_tax,
                "broker_fee": args.broker_fee,
            },
            "industry": {
                "mfg_system_id": args.mfg_system_id,
                "rx_system_id": args.rx_system_id,
                "inv_system_id": args.inv_system_id,
                "mfg_cost_index": mfg_cost_index,
                "rx_cost_index": rx_cost_index,
                "inv_cost_index": inv_cost_index,
                "facility_tax": args.facility_tax,
                "structure_job_bonus": args.structure_job_bonus,
            },
            "depth": {
                "top_n": args.depth_top_n,
                "max_materials": args.depth_max_materials,
                "input_slippage": args.input_slippage,
                "output_slippage": args.output_slippage,
                "safety": args.depth_safety,
                "note": "Depth uses ESI region orderbooks (structure markets may be undercounted).",
            },
        },
        "manufacturing": manufacturing_rows,
        "reactions": reaction_rows,
        "refining": refining_rows,
        "t2": t2_rows,
    }

    out_path = Path(args.out)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Write daily history snapshot (optional but awesome for trend charts)
    hist_dir = Path("docs/data/history")
    ensure_parent(hist_dir / "x")
    hist_path = hist_dir / f"{utc_date_str()}.json"
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass

    print(f"[update_rankings] Wrote {out_path} (and history snapshot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
