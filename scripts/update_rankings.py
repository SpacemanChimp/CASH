#!/usr/bin/env python3
"""
update_rankings.py

Reads recipe data (data/recipes.json.gz), fetches market prices, computes rankings,
and writes docs/data/rankings.json for GitHub Pages.

Price sources:
- Fuzzwork market aggregates CSV dump (region aggregates): https://market.fuzzwork.co.uk/aggregatecsv.csv.gz
  (updated about every 30 minutes, contains "5% average" aka fivepercent).
- Optional station-level aggregates for shortlisted candidates:
  https://market.fuzzwork.co.uk/aggregates/?station=60003760&types=34,35,...

Notes:
- This is a *decision-support tool*. It does not include:
  - industry job installation fees (system cost index)
  - hauling, risk, liquidity/volume constraints
  - structure/rig bonuses, skills, ME/TE rounding
  Adjust assumptions in code or extend the project for personalization via ESI.
"""

from __future__ import annotations

import argparse
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


AGGREGATECSV_URL = "https://market.fuzzwork.co.uk/aggregatecsv.csv.gz"

DEFAULT_REGION_ID = 10000002  # The Forge (Jita's region)
DEFAULT_STATION_ID = 60003760  # Jita IV - Moon 4 - Caldari Navy Assembly Plant

RECIPES_PATH = Path("data/recipes.json.gz")
RANKINGS_PATH = Path("docs/data/rankings.json")

# When using station-level, query in chunks to avoid URL limits.
STATION_CHUNK = 200


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_recipes(recipes_path: Path) -> Dict[str, Any]:
    with gzip.open(recipes_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def maybe_build_recipes(recipes_path: Path, force: bool = False) -> None:
    if recipes_path.exists() and not force:
        return
    # Run build_recipes.py
    print("[update_rankings] recipes.json.gz missing; building from SDE (one-time-ish)…")
    import subprocess, sys
    cmd = [sys.executable, "scripts/build_recipes.py", "--out", str(recipes_path)]
    if force:
        cmd.append("--force")
    subprocess.check_call(cmd)


def download_to(url: str, dest: Path, timeout: int = 60) -> Dict[str, str]:
    ensure_parent(dest)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        headers = {k: v for k, v in r.headers.items()}
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return headers


def load_region_prices_from_aggregatecsv(cache_path: Path, region_id: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, str]]:
    """
    Returns:
      prices[type_id] = {"buy": {...}, "sell": {...}}
      headers (HTTP response headers from download if we downloaded)
    """
    headers: Dict[str, str] = {}
    if not cache_path.exists():
        print(f"[update_rankings] Downloading region aggregates CSV: {AGGREGATECSV_URL}")
        headers = download_to(AGGREGATECSV_URL, cache_path)

    prices: Dict[int, Dict[str, Any]] = {}
    with gzip.open(cache_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns like:
        # what, weightedaverage, maxval, minval, stddev, median, volume, numorders, fivepercent, orderSet
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


def fuzz_price(stats: Dict[str, Any], side: str, mode: str) -> float:
    """
    side: "buy" or "sell"
    mode:
      - "minmax": buy=max buy, sell=min sell (fast execution)
      - "percentile": use 5% average (percentile) when available, else fallback
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


def fetch_station_aggregates(station_id: int, type_ids: List[int], sleep_s: float = 0.15) -> Dict[int, Dict[str, Any]]:
    """
    Fetches station-level aggregates from market.fuzzwork.co.uk for a list of typeIDs.
    Returns dict keyed by type_id int, value includes "buy"/"sell" dicts.
    """
    out: Dict[int, Dict[str, Any]] = {}
    base = f"https://market.fuzzwork.co.uk/aggregates/?station={station_id}&types="

    for i in range(0, len(type_ids), STATION_CHUNK):
        chunk = type_ids[i : i + STATION_CHUNK]
        url = base + ",".join(str(x) for x in chunk)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        # keys are typeIDs as strings
        for k, v in data.items():
            try:
                tid = int(k)
            except ValueError:
                continue
            out[tid] = v
        time.sleep(sleep_s)
    return out


def compute_entry(
    *,
    recipe: Dict[str, Any],
    prices: Dict[int, Dict[str, Any]],
    price_mode: str,
    fee_rate: float,
    material_multiplier: float = 1.0,
    is_refining: bool = False,
    reprocess_yield: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    Returns a normalized entry dict with:
      cost, revenue, fees, profit, roi, time_s, profit_per_hour (if time_s>0)
      + materials breakdown for UI
    """
    if is_refining:
        inp_id = int(recipe["input_type_id"])
        batch_units = int(recipe.get("portion_size") or 1)
        unit_vol = float(recipe.get("volume") or 0.0)
        batch_m3 = unit_vol * batch_units

        inp_stats = prices.get(inp_id, {})
        inp_price = fuzz_price(inp_stats, "sell", price_mode)
        if inp_price <= 0:
            return None
        cost = inp_price * batch_units

        outputs = recipe.get("outputs") or []
        mats = []
        revenue = 0.0
        for o in outputs:
            out_id = int(o["type_id"])
            qty = float(o["qty"] or 0.0) * float(reprocess_yield)
            out_stats = prices.get(out_id, {})
            out_price = fuzz_price(out_stats, "buy", price_mode)
            if out_price <= 0:
                # If output isn't priced, skip this recipe entirely (very likely junk)
                return None
            ext = out_price * qty
            revenue += ext
            mats.append({
                "type_id": out_id,
                "name": o.get("name", f"type {out_id}"),
                "qty": round(qty, 3),
                "unit_price": out_price,
                "extended": ext,
            })

        fees = revenue * fee_rate
        profit = revenue - fees - cost
        roi = (profit / cost) if cost > 0 else None
        profit_per_m3 = (profit / batch_m3) if batch_m3 > 0 else None

        return {
            "kind": "refining",
            "input_type_id": inp_id,
            "input_name": recipe.get("input_name", f"type {inp_id}"),
            "batch_units": batch_units,
            "batch_m3": batch_m3,
            "time_s": 0,
            "cost": cost,
            "revenue": revenue,
            "fees": fees,
            "profit": profit,
            "roi": roi,
            "profit_per_m3": profit_per_m3,
            "materials": mats,  # output breakdown
            "blueprint_cost": None,
            "payback_runs": None,
        }

    # Industry (manufacturing / reactions)
    bp_id = int(recipe["blueprint_type_id"])
    prod_id = int(recipe["product_type_id"])
    pqty = float(recipe.get("product_qty") or 1.0)
    time_s = int(recipe.get("time_s") or 0)

    # Inputs
    mats = []
    cost = 0.0
    for m in recipe.get("materials") or []:
        tid = int(m["type_id"])
        qty = float(m["qty"] or 0.0) * float(material_multiplier)
        stats = prices.get(tid, {})
        unit_price = fuzz_price(stats, "sell", price_mode)
        if unit_price <= 0:
            return None
        ext = unit_price * qty
        cost += ext
        mats.append({
            "type_id": tid,
            "name": m.get("name", f"type {tid}"),
            "qty": round(qty, 3),
            "unit_price": unit_price,
            "extended": ext,
        })

    # Output
    out_stats = prices.get(prod_id, {})
    out_price = fuzz_price(out_stats, "buy", price_mode)
    if out_price <= 0:
        return None
    revenue = out_price * pqty

    fees = revenue * fee_rate
    profit = revenue - fees - cost
    roi = (profit / cost) if cost > 0 else None
    profit_per_hour = (profit / time_s * 3600.0) if time_s > 0 else None

    # Blueprint acquisition cost (approximate: market sell price for the blueprint item)
    bp_stats = prices.get(bp_id, {})
    bp_cost = fuzz_price(bp_stats, "sell", price_mode)
    payback_runs = (bp_cost / profit) if (bp_cost > 0 and profit > 0) else None

    return {
        "kind": recipe.get("kind", "industry"),
        "blueprint_type_id": bp_id,
        "blueprint_name": recipe.get("blueprint_name", f"type {bp_id}"),
        "product_type_id": prod_id,
        "product_name": recipe.get("product_name", f"type {prod_id}"),
        "product_qty": pqty,
        "time_s": time_s,
        "cost": cost,
        "revenue": revenue,
        "fees": fees,
        "profit": profit,
        "roi": roi,
        "profit_per_hour": profit_per_hour,
        "materials": mats,
        "blueprint_cost": bp_cost if bp_cost > 0 else None,
        "payback_runs": payback_runs,
        "profit_per_m3": None,
    }


def unique_type_ids_from_entries(entries: List[Dict[str, Any]]) -> Set[int]:
    ids: Set[int] = set()
    for e in entries:
        if e.get("blueprint_type_id"):
            ids.add(int(e["blueprint_type_id"]))
        if e.get("product_type_id"):
            ids.add(int(e["product_type_id"]))
        if e.get("input_type_id"):
            ids.add(int(e["input_type_id"]))
        for m in e.get("materials") or []:
            ids.add(int(m["type_id"]))
    return ids


def recompute_with_prices(
    recipes: List[Dict[str, Any]],
    prices: Dict[int, Dict[str, Any]],
    *,
    kind: str,
    price_mode: str,
    fee_rate: float,
    material_multiplier: float,
    reprocess_yield: float,
) -> List[Dict[str, Any]]:
    out = []
    for r in recipes:
        if kind == "refining":
            e = compute_entry(
                recipe=r,
                prices=prices,
                price_mode=price_mode,
                fee_rate=fee_rate,
                is_refining=True,
                reprocess_yield=reprocess_yield,
            )
        else:
            e = compute_entry(
                recipe=r,
                prices=prices,
                price_mode=price_mode,
                fee_rate=fee_rate,
                material_multiplier=material_multiplier,
                is_refining=False,
            )
            if e:
                e["kind"] = kind
        if e:
            out.append(e)
    return out


def top_n(entries: List[Dict[str, Any]], *, by: str, n: int) -> List[Dict[str, Any]]:
    def key(e: Dict[str, Any]) -> float:
        v = e.get(by)
        try:
            return float(v) if v is not None else float("-inf")
        except Exception:
            return float("-inf")
    return sorted(entries, key=key, reverse=True)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", type=int, default=DEFAULT_REGION_ID)
    ap.add_argument("--station", type=int, default=DEFAULT_STATION_ID)
    ap.add_argument("--price-mode", choices=["minmax", "percentile", "weighted"], default="percentile")
    ap.add_argument("--fee-rate", type=float, default=0.03)
    ap.add_argument("--material-multiplier", type=float, default=1.0)
    ap.add_argument("--reprocess-yield", type=float, default=0.7)
    ap.add_argument("--candidates", type=int, default=250, help="Shortlist size per category before station refresh.")
    ap.add_argument("--top", type=int, default=100, help="Final rows per category written to rankings.json.")
    ap.add_argument("--force-recipes", action="store_true", help="Force rebuild recipes from SDE.")
    ap.add_argument("--cache-dir", default=".cache")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    maybe_build_recipes(RECIPES_PATH, force=args.force_recipes)
    recipes = load_recipes(RECIPES_PATH)

    # 1) Fast pass: region prices from aggregatecsv
    agg_cache = cache_dir / "aggregatecsv.csv.gz"
    region_prices, headers = load_region_prices_from_aggregatecsv(agg_cache, args.region)

    # Helper: merge station prices over region prices
    def merged_prices(station_prices: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        merged = dict(region_prices)
        for tid, v in station_prices.items():
            merged[tid] = v
        return merged

    # 2) Compute region-based ranks for all recipes
    mfg_entries_region = recompute_with_prices(
        recipes["manufacturing"],
        region_prices,
        kind="manufacturing",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )
    rx_entries_region = recompute_with_prices(
        recipes["reactions"],
        region_prices,
        kind="reactions",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )
    ref_entries_region = recompute_with_prices(
        recipes["refining"],
        region_prices,
        kind="refining",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )

    # 3) Shortlist candidates (region pass)
    mfg_candidates = top_n(mfg_entries_region, by="profit_per_hour", n=args.candidates)
    rx_candidates = top_n(rx_entries_region, by="profit_per_hour", n=args.candidates)
    ref_candidates = top_n(ref_entries_region, by="profit_per_m3", n=args.candidates)

    # 4) Fetch station prices for types involved in shortlisted candidates
    needed_ids = sorted(list(
        unique_type_ids_from_entries(mfg_candidates)
        | unique_type_ids_from_entries(rx_candidates)
        | unique_type_ids_from_entries(ref_candidates)
    ))

    station_prices: Dict[int, Dict[str, Any]] = {}
    if needed_ids:
        print(f"[update_rankings] Fetching station aggregates for {len(needed_ids)} typeIDs (station={args.station})…")
        try:
            station_prices = fetch_station_aggregates(args.station, needed_ids)
        except Exception as e:
            print(f"[update_rankings] WARNING: station aggregates fetch failed; using region-only prices. Error: {e}")
            station_prices = {}

    all_prices = merged_prices(station_prices) if station_prices else region_prices

    # 5) Recompute candidates with (station-over-region) prices, then choose final top N
    # To recompute, we need the original recipes for those candidates, not the entries.
    # We'll map by blueprint+product for industry, and input_type for refining.
    def index_recipes_industry(rs: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
        idx = {}
        for r in rs:
            idx[(int(r["blueprint_type_id"]), int(r["product_type_id"]))] = r
        return idx

    mfg_idx = index_recipes_industry(recipes["manufacturing"])
    rx_idx = index_recipes_industry(recipes["reactions"])
    ref_idx = {int(r["input_type_id"]): r for r in recipes["refining"]}

    mfg_recipes_short = [mfg_idx[(int(e["blueprint_type_id"]), int(e["product_type_id"]))] for e in mfg_candidates if (int(e["blueprint_type_id"]), int(e["product_type_id"])) in mfg_idx]
    rx_recipes_short = [rx_idx[(int(e["blueprint_type_id"]), int(e["product_type_id"]))] for e in rx_candidates if (int(e["blueprint_type_id"]), int(e["product_type_id"])) in rx_idx]
    ref_recipes_short = [ref_idx[int(e["input_type_id"])] for e in ref_candidates if int(e["input_type_id"]) in ref_idx]

    mfg_entries = recompute_with_prices(
        mfg_recipes_short,
        all_prices,
        kind="manufacturing",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )
    rx_entries = recompute_with_prices(
        rx_recipes_short,
        all_prices,
        kind="reactions",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )
    ref_entries = recompute_with_prices(
        ref_recipes_short,
        all_prices,
        kind="refining",
        price_mode=args.price_mode,
        fee_rate=args.fee_rate,
        material_multiplier=args.material_multiplier,
        reprocess_yield=args.reprocess_yield,
    )

    # Final ranking
    mfg_top = top_n(mfg_entries, by="profit_per_hour", n=args.top)
    rx_top = top_n(rx_entries, by="profit_per_hour", n=args.top)
    ref_top = top_n(ref_entries, by="profit_per_m3", n=args.top)

    out = {
        "generated_at": utc_now_iso(),
        "market": {
            "region_id": args.region,
            "station_id": args.station,
            "price_sources": {
                "aggregatecsv_url": AGGREGATECSV_URL,
                "station_aggregates": f"https://market.fuzzwork.co.uk/aggregates/?station={args.station}&types=…",
            },
            "aggregatecsv_http_headers": headers,
        },
        "assumptions": {
            "price_mode": args.price_mode,
            "fee_rate": args.fee_rate,
            "material_multiplier": args.material_multiplier,
            "reprocess_yield": args.reprocess_yield,
            "notes": [
                "Input prices use 'sell' side; output prices use 'buy' side (quick execution model).",
                "Blueprint costs are approximated from blueprint item sell prices (if available).",
                "Industry installation costs, structure bonuses, and ME/TE rounding are not included by default.",
            ],
        },
        "manufacturing": mfg_top,
        "reactions": rx_top,
        "refining": ref_top,
        "debug": {
            "counts": {
                "manufacturing_total_recipes": len(recipes.get("manufacturing", [])),
                "reactions_total_recipes": len(recipes.get("reactions", [])),
                "refining_total_inputs": len(recipes.get("refining", [])),
                "manufacturing_region_scored": len(mfg_entries_region),
                "reactions_region_scored": len(rx_entries_region),
                "refining_region_scored": len(ref_entries_region),
                "station_types_fetched": len(station_prices),
                "manufacturing_candidates": len(mfg_candidates),
                "reactions_candidates": len(rx_candidates),
                "refining_candidates": len(ref_candidates),
            }
        }
    }

    ensure_parent(RANKINGS_PATH)
    with open(RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[update_rankings] Wrote {RANKINGS_PATH}")
    print(f"[update_rankings] Top manufacturing profit/hour: {mfg_top[0]['product_name'] if mfg_top else 'n/a'}")
    print(f"[update_rankings] Top reactions profit/hour: {rx_top[0]['product_name'] if rx_top else 'n/a'}")
    print(f"[update_rankings] Top refining profit/m3: {ref_top[0]['input_name'] if ref_top else 'n/a'}")


if __name__ == "__main__":
    main()
