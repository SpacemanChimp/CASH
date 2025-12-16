#!/usr/bin/env python3
"""
build_recipes.py

Builds a compact recipes JSON (gzipped) from the EVE SDE SQLite (Fuzzwork eve.db),
covering:
  - manufacturing (industry activityID = 1)
  - reactions      (industry activityID = 11)
  - refining / reprocessing (invTypeMaterials, ore/ice-like items)

Output schema (what update_rankings.py expects):
{
  "generated_at": "...",
  "types": {
     "<type_id>": {"name": "...", "volume": 0.0, "portionSize": 1}
  },
  "manufacturing": [
     {
       "blueprint_type_id": 123,
       "blueprint_name": "Some Blueprint",
       "product_type_id": 456,
       "product_name": "Some Item",
       "output_qty": 1,
       "time_s": 600,
       "materials": [{"type_id": 34, "name": "Tritanium", "qty": 1000}, ...]
     }, ...
  ],
  "reactions": [ ... same shape ... ],
  "refining": [
     {
       "input_type_id": 1230,
       "input_name": "Veldspar",
       "batch_units": 333,
       "batch_m3": 33.3,
       "outputs": [{"type_id": 34, "name": "Tritanium", "qty": 1000}, ...]
     }, ...
  ]
}

Notes:
- This is "static recipe" data. Profitability comes from update_rankings.py using market prices.
- The filtering for "ore/ice-like" inputs is heuristic, based on group/category names containing
  "asteroid", "ore", or "ice". If CCP renames categories, you can loosen/tighten the filter below.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

EVE_DB_BZ2_URL = "https://www.fuzzwork.co.uk/dump/latest/eve.db.bz2"
CACHE_DIR = Path(".cache")
EVE_DB_BZ2_PATH = CACHE_DIR / "eve.db.bz2"
EVE_DB_PATH = CACHE_DIR / "eve.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, timeout: int = 180) -> None:
    ensure_parent(dest)
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "eve-money-button/1.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_eve_db(force: bool = False) -> None:
    """
    Ensures .cache/eve.db exists (decompressed).
    Downloads eve.db.bz2 if needed.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if force or not EVE_DB_BZ2_PATH.exists():
        print(f"[build_recipes] Downloading {EVE_DB_BZ2_URL}")
        download(EVE_DB_BZ2_URL, EVE_DB_BZ2_PATH)

    if force or not EVE_DB_PATH.exists():
        print("[build_recipes] Decompressing eve.db.bz2 → eve.db")
        # Decompress (bz2) to a sqlite file
        with bz2.open(EVE_DB_BZ2_PATH, "rb") as src, open(EVE_DB_PATH, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)


def load_types(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping typeID -> {name, volume, portionSize}
    """
    types: Dict[int, Dict[str, Any]] = {}
    # invTypes columns are stable: typeID, typeName, volume, portionSize, published
    cur = conn.execute(
        "SELECT typeID, typeName, COALESCE(volume, 0), COALESCE(portionSize, 1) "
        "FROM invTypes "
        "WHERE COALESCE(published, 1) = 1"
    )
    for type_id, name, vol, portion in cur.fetchall():
        try:
            tid = int(type_id)
        except Exception:
            continue
        types[tid] = {
            "name": str(name),
            "volume": float(vol) if vol is not None else 0.0,
            "portionSize": int(portion) if portion is not None else 1,
        }
    return types


def build_industry_recipes(conn: sqlite3.Connection, types: Dict[int, Dict[str, Any]], activity_id: int) -> List[Dict[str, Any]]:
    """
    Builds manufacturing or reaction recipes based on industryActivity* tables.
    """
    # Pull activity rows with products in one go
    cur = conn.execute(
        "SELECT ia.blueprintTypeID, ia.time, p.productTypeID, p.quantity "
        "FROM industryActivity ia "
        "JOIN industryActivityProducts p "
        "  ON ia.blueprintTypeID = p.blueprintTypeID AND ia.activityID = p.activityID "
        "WHERE ia.activityID = ?",
        (activity_id,),
    )

    # Choose the primary product per blueprint: highest quantity
    best_product: Dict[Tuple[int, int], Tuple[int, float]] = {}  # (bp, activity) -> (prod, qty)
    time_s: Dict[Tuple[int, int], int] = {}

    for bp_id, t, prod_id, qty in cur.fetchall():
        key = (int(bp_id), activity_id)
        time_s[key] = int(t) if t is not None else 0
        prod = int(prod_id)
        q = float(qty) if qty is not None else 1.0
        prev = best_product.get(key)
        if prev is None or q > prev[1]:
            best_product[key] = (prod, q)

    # Materials lookup prepared statement
    mats_stmt = "SELECT materialTypeID, quantity FROM industryActivityMaterials WHERE blueprintTypeID = ? AND activityID = ?"

    out: List[Dict[str, Any]] = []
    for (bp, act), (prod, out_qty) in best_product.items():
        # Materials
        mats: List[Dict[str, Any]] = []
        for mtid, mqty in conn.execute(mats_stmt, (bp, act)).fetchall():
            mt = int(mtid)
            q = float(mqty) if mqty is not None else 0.0
            if q <= 0:
                continue
            mats.append({"type_id": mt, "name": types.get(mt, {}).get("name", f"type {mt}"), "qty": q})

        if not mats:
            continue

        bp_name = types.get(bp, {}).get("name", f"type {bp}")
        prod_name = types.get(prod, {}).get("name", f"type {prod}")
        out.append(
            {
                "blueprint_type_id": bp,
                "blueprint_name": bp_name,
                "product_type_id": prod,
                "product_name": prod_name,
                "output_qty": float(out_qty) if out_qty and out_qty > 0 else 1.0,
                "time_s": int(time_s.get((bp, act), 0)),
                "materials": mats,
            }
        )

    return out


def build_refining_recipes(conn: sqlite3.Connection, types: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds reprocessing recipes from invTypeMaterials.
    Attempts to restrict to ore/ice-like items by group/category names.
    """
    # Heuristic selection of candidate inputs with reprocessing yields.
    # We join invGroups + invCategories to look at names.
    cur = conn.execute(
        "SELECT t.typeID, t.typeName, COALESCE(t.volume, 0), COALESCE(t.portionSize, 1), "
        "       COALESCE(g.groupName, ''), COALESCE(c.categoryName, '') "
        "FROM invTypes t "
        "JOIN invGroups g ON t.groupID = g.groupID "
        "JOIN invCategories c ON g.categoryID = c.categoryID "
        "WHERE COALESCE(t.published, 1) = 1 "
        "  AND t.typeID IN (SELECT DISTINCT typeID FROM invTypeMaterials)"
    )

    def looks_like_ore_or_ice(group_name: str, cat_name: str, type_name: str) -> bool:
        s = f"{group_name} {cat_name} {type_name}".lower()
        return ("asteroid" in s) or (" ore" in s) or s.endswith("ore") or ("ice" in s)

    candidates: List[Tuple[int, str, float, int]] = []
    for tid, tname, vol, portion, gname, cname in cur.fetchall():
        tid = int(tid)
        tname = str(tname)
        gname = str(gname)
        cname = str(cname)
        if not looks_like_ore_or_ice(gname, cname, tname):
            continue
        portion_i = int(portion) if portion is not None else 1
        if portion_i <= 0:
            portion_i = 1
        vol_f = float(vol) if vol is not None else 0.0
        candidates.append((tid, tname, vol_f, portion_i))

    # Now build outputs for each candidate
    out: List[Dict[str, Any]] = []
    stmt = "SELECT materialTypeID, quantity FROM invTypeMaterials WHERE typeID = ?"
    for tid, tname, vol, portion in candidates:
        outs: List[Dict[str, Any]] = []
        for mtid, qty in conn.execute(stmt, (tid,)).fetchall():
            mt = int(mtid)
            q = float(qty) if qty is not None else 0.0
            if q <= 0:
                continue
            outs.append({"type_id": mt, "name": types.get(mt, {}).get("name", f"type {mt}"), "qty": q})

        if not outs:
            continue

        batch_units = portion
        batch_m3 = float(batch_units) * float(vol)

        out.append(
            {
                "input_type_id": tid,
                "input_name": tname,
                "batch_units": int(batch_units),
                "batch_m3": float(batch_m3),
                "outputs": outs,
            }
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/recipes.json.gz", help="Output path for recipes.json.gz")
    ap.add_argument("--force", action="store_true", help="Force re-download / re-decompress eve.db")
    args = ap.parse_args()

    out_path = Path(args.out)

    ensure_eve_db(force=args.force)

    print(f"[build_recipes] Opening SQLite: {EVE_DB_PATH}")
    conn = sqlite3.connect(str(EVE_DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        types = load_types(conn)
        print(f"[build_recipes] Loaded {len(types)} published types")

        manufacturing = build_industry_recipes(conn, types, activity_id=1)
        print(f"[build_recipes] Manufacturing recipes: {len(manufacturing)}")

        reactions = build_industry_recipes(conn, types, activity_id=11)
        print(f"[build_recipes] Reaction recipes: {len(reactions)}")

        refining = build_refining_recipes(conn, types)
        print(f"[build_recipes] Refining recipes: {len(refining)}")

    finally:
        conn.close()

    payload = {
        "generated_at": utc_now_iso(),
        "types": {str(k): v for k, v in types.items()},
        "manufacturing": manufacturing,
        "reactions": reactions,
        "refining": refining,
    }

    ensure_parent(out_path)
    print(f"[build_recipes] Writing: {out_path}")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
