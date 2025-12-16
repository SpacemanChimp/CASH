#!/usr/bin/env python3
"""
build_recipes.py

Builds a compact recipes JSON (gzipped) from the EVE SDE SQLite (Fuzzwork eve.db),
covering:
  - manufacturing (industry activityID = 1)
  - reactions      (industry activityID = 11)
  - refining / reprocessing (invTypeMaterials, ore/ice-like items)

This version auto-detects column names in the SQLite, because Fuzzwork's eve.db
schema occasionally differs from the canonical SDE names (e.g. blueprintTypeID
sometimes appears as typeID).

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
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import sqlite3
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
        with bz2.open(EVE_DB_BZ2_PATH, "rb") as src, open(EVE_DB_PATH, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols: List[str] = []
    for r in rows:
        # sqlite3.Row supports both index and key access
        name = r["name"] if isinstance(r, sqlite3.Row) else r[1]
        cols.append(str(name))
    return cols


def pick_col(conn: sqlite3.Connection, table: str, candidates: List[str]) -> str:
    cols = table_columns(conn, table)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c:
            return c
    raise RuntimeError(f"[build_recipes] Could not find expected column in table={table}. Have columns: {cols}")


def load_types(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping typeID -> {name, volume, portionSize}
    """
    t_typeid = pick_col(conn, "invTypes", ["typeID"])
    t_typename = pick_col(conn, "invTypes", ["typeName", "typeNAME", "name"])
    t_volume = pick_col(conn, "invTypes", ["volume"])
    t_portion = pick_col(conn, "invTypes", ["portionSize", "portion"])
    t_published = pick_col(conn, "invTypes", ["published"])

    types: Dict[int, Dict[str, Any]] = {}
    cur = conn.execute(
        f"SELECT {t_typeid} as typeID, {t_typename} as typeName, COALESCE({t_volume}, 0) as volume, COALESCE({t_portion}, 1) as portionSize "
        f"FROM invTypes WHERE COALESCE({t_published}, 1) = 1"
    )
    for row in cur.fetchall():
        tid = int(row["typeID"])
        types[tid] = {
            "name": str(row["typeName"]),
            "volume": float(row["volume"]) if row["volume"] is not None else 0.0,
            "portionSize": int(row["portionSize"]) if row["portionSize"] is not None else 1,
        }
    return types


def build_industry_recipes(conn: sqlite3.Connection, types: Dict[int, Dict[str, Any]], activity_id: int) -> List[Dict[str, Any]]:
    """
    Builds manufacturing or reaction recipes based on industryActivity* tables.
    Auto-detects column names (blueprintTypeID sometimes appears as typeID).
    """
    ia_bp = pick_col(conn, "industryActivity", ["blueprintTypeID", "typeID"])
    ia_act = pick_col(conn, "industryActivity", ["activityID"])
    ia_time = pick_col(conn, "industryActivity", ["time", "duration"])

    iap_bp = pick_col(conn, "industryActivityProducts", ["blueprintTypeID", "typeID"])
    iap_act = pick_col(conn, "industryActivityProducts", ["activityID"])
    iap_prod = pick_col(conn, "industryActivityProducts", ["productTypeID"])
    iap_qty = pick_col(conn, "industryActivityProducts", ["quantity", "qty"])

    iam_bp = pick_col(conn, "industryActivityMaterials", ["blueprintTypeID", "typeID"])
    iam_act = pick_col(conn, "industryActivityMaterials", ["activityID"])
    iam_mat = pick_col(conn, "industryActivityMaterials", ["materialTypeID"])
    iam_qty = pick_col(conn, "industryActivityMaterials", ["quantity", "qty"])

    cur = conn.execute(
        f"SELECT ia.{ia_bp} AS bp, ia.{ia_time} AS time_s, p.{iap_prod} AS prod, p.{iap_qty} AS qty "
        f"FROM industryActivity ia "
        f"JOIN industryActivityProducts p "
        f"  ON ia.{ia_bp} = p.{iap_bp} AND ia.{ia_act} = p.{iap_act} "
        f"WHERE ia.{ia_act} = ?",
        (activity_id,),
    )

    best_product: Dict[int, Tuple[int, float]] = {}  # bp -> (prod, qty)
    time_s: Dict[int, int] = {}  # bp -> time

    for row in cur.fetchall():
        bp = int(row["bp"])
        t = int(row["time_s"]) if row["time_s"] is not None else 0
        prod = int(row["prod"])
        qty = float(row["qty"]) if row["qty"] is not None else 1.0

        time_s[bp] = t
        prev = best_product.get(bp)
        if prev is None or qty > prev[1]:
            best_product[bp] = (prod, qty)

    mats_stmt = (
        f"SELECT {iam_mat} AS mat, {iam_qty} AS qty "
        f"FROM industryActivityMaterials WHERE {iam_bp} = ? AND {iam_act} = ?"
    )

    out: List[Dict[str, Any]] = []
    for bp, (prod, out_qty) in best_product.items():
        mats: List[Dict[str, Any]] = []
        for mrow in conn.execute(mats_stmt, (bp, activity_id)).fetchall():
            mt = int(mrow["mat"])
            q = float(mrow["qty"]) if mrow["qty"] is not None else 0.0
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
                "time_s": int(time_s.get(bp, 0)),
                "materials": mats,
            }
        )

    return out


def build_refining_recipes(conn: sqlite3.Connection, types: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds reprocessing recipes from invTypeMaterials for ore/ice-like items.
    """
    t_typeid = pick_col(conn, "invTypes", ["typeID"])
    t_typename = pick_col(conn, "invTypes", ["typeName", "name"])
    t_volume = pick_col(conn, "invTypes", ["volume"])
    t_portion = pick_col(conn, "invTypes", ["portionSize", "portion"])
    t_groupid = pick_col(conn, "invTypes", ["groupID"])
    t_published = pick_col(conn, "invTypes", ["published"])

    g_groupid = pick_col(conn, "invGroups", ["groupID"])
    g_groupname = pick_col(conn, "invGroups", ["groupName", "name"])
    g_catid = pick_col(conn, "invGroups", ["categoryID"])

    c_catid = pick_col(conn, "invCategories", ["categoryID"])
    c_catname = pick_col(conn, "invCategories", ["categoryName", "name"])

    itm_typeid = pick_col(conn, "invTypeMaterials", ["typeID"])
    itm_mat = pick_col(conn, "invTypeMaterials", ["materialTypeID"])
    itm_qty = pick_col(conn, "invTypeMaterials", ["quantity", "qty"])

    cur = conn.execute(
        f"SELECT t.{t_typeid} AS typeID, t.{t_typename} AS typeName, COALESCE(t.{t_volume}, 0) AS volume, "
        f"       COALESCE(t.{t_portion}, 1) AS portionSize, "
        f"       COALESCE(g.{g_groupname}, '') AS groupName, COALESCE(c.{c_catname}, '') AS categoryName "
        f"FROM invTypes t "
        f"JOIN invGroups g ON t.{t_groupid} = g.{g_groupid} "
        f"JOIN invCategories c ON g.{g_catid} = c.{c_catid} "
        f"WHERE COALESCE(t.{t_published}, 1) = 1 "
        f"  AND t.{t_typeid} IN (SELECT DISTINCT {itm_typeid} FROM invTypeMaterials)"
    )

    def looks_like_ore_or_ice(group_name: str, cat_name: str, type_name: str) -> bool:
        s = f"{group_name} {cat_name} {type_name}".lower()
        # Broad-ish. We want ore & ice, not "everything that can be reprocessed".
        if "asteroid" in s:
            return True
        if "ice" in s:
            return True
        if "ore" in s:
            return True
        return False

    candidates: List[Tuple[int, str, float, int]] = []
    for row in cur.fetchall():
        tid = int(row["typeID"])
        tname = str(row["typeName"])
        vol = float(row["volume"]) if row["volume"] is not None else 0.0
        portion = int(row["portionSize"]) if row["portionSize"] is not None else 1
        gname = str(row["groupName"])
        cname = str(row["categoryName"])

        if not looks_like_ore_or_ice(gname, cname, tname):
            continue
        if portion <= 0:
            portion = 1

        candidates.append((tid, tname, vol, portion))

    stmt = f"SELECT {itm_mat} AS mat, {itm_qty} AS qty FROM invTypeMaterials WHERE {itm_typeid} = ?"

    out: List[Dict[str, Any]] = []
    for tid, tname, vol, portion in candidates:
        outs: List[Dict[str, Any]] = []
        for mrow in conn.execute(stmt, (tid,)).fetchall():
            mt = int(mrow["mat"])
            q = float(mrow["qty"]) if mrow["qty"] is not None else 0.0
            if q <= 0:
                continue
            outs.append({"type_id": mt, "name": types.get(mt, {}).get("name", f"type {mt}"), "qty": q})

        if not outs:
            continue

        batch_units = int(portion)
        batch_m3 = float(batch_units) * float(vol)

        out.append(
            {
                "input_type_id": tid,
                "input_name": tname,
                "batch_units": batch_units,
                "batch_m3": batch_m3,
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
