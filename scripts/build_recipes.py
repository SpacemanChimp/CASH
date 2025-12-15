#!/usr/bin/env python3
"""
build_recipes.py

Builds a compact recipes file from the EVE SDE (via Fuzzwork's SQLite conversion),
so that routine price updates don't need to re-download the entire SDE.

Sources:
- Fuzzwork SDE SQLite conversion is published at https://www.fuzzwork.co.uk/dump/latest/eve.db.bz2 (see EVE forums post by Steve Ronuken).
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

FUZZWORK_SDE_SQLITE_BZ2 = "https://www.fuzzwork.co.uk/dump/latest/eve.db.bz2"

ACTIVITY_MANUFACTURING = 1
ACTIVITY_REACTIONS = 11


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, timeout: int = 60) -> None:
    _ensure_parent(dest)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def decompress_bz2(src: Path, dest: Path) -> None:
    _ensure_parent(dest)
    with bz2.open(src, "rb") as fin, open(dest, "wb") as fout:
        while True:
            b = fin.read(1024 * 1024)
            if not b:
                break
            fout.write(b)


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]  # name column


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return r is not None


def pick_col(cols: List[str], *candidates: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found in columns={cols}")


def build_recipes(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Basic sanity checks
    for t in ["invTypes", "industryActivityMaterials", "industryActivityProducts", "invTypeMaterials"]:
        if not has_table(conn, t):
            raise RuntimeError(f"Expected table '{t}' not found in {db_path}")

    inv_cols = table_columns(conn, "invTypes")
    c_type_id = pick_col(inv_cols, "typeID")
    c_type_name = pick_col(inv_cols, "typeName")
    c_volume = pick_col(inv_cols, "volume")
    c_portion = pick_col(inv_cols, "portionSize")
    c_published = pick_col(inv_cols, "published")
    # marketGroupID is useful for filtering, but not always present in older conversions.
    c_mktgrp = "marketGroupID" if "marketGroupID" in inv_cols else None

    # Pull all type names/volumes we might reference.
    # (We avoid loading *everything* into RAM if possible, but in practice this is fine.)
    type_rows = conn.execute(
        f"SELECT {c_type_id} AS type_id, {c_type_name} AS type_name, {c_volume} AS volume, {c_portion} AS portion, {c_published} AS published"
        + (f", {c_mktgrp} AS market_group" if c_mktgrp else "")
        + " FROM invTypes;"
    ).fetchall()

    types: Dict[int, Dict[str, Any]] = {}
    for r in type_rows:
        tid = int(r["type_id"])
        types[tid] = {
            "name": r["type_name"],
            "volume": float(r["volume"] or 0.0),
            "portion": int(r["portion"] or 1),
            "published": int(r["published"] or 0),
            "market_group": int(r["market_group"]) if (c_mktgrp and r["market_group"] is not None) else None,
        }

    # Activity time (seconds) per blueprint per activity
    activity_time: Dict[Tuple[int, int], int] = {}
    if has_table(conn, "industryActivity"):
        ia_cols = table_columns(conn, "industryActivity")
        c_ia_type = pick_col(ia_cols, "typeID")
        c_ia_act = pick_col(ia_cols, "activityID")
        c_ia_time = pick_col(ia_cols, "time")
        for r in conn.execute(
            f"SELECT {c_ia_type} AS type_id, {c_ia_act} AS act_id, {c_ia_time} AS time_s FROM industryActivity;"
        ).fetchall():
            activity_time[(int(r["type_id"]), int(r["act_id"]))] = int(r["time_s"] or 0)

    def build_industry_recipes(activity_id: int) -> List[Dict[str, Any]]:
        # Materials
        iam_cols = table_columns(conn, "industryActivityMaterials")
        c_bp = pick_col(iam_cols, "typeID")
        c_act = pick_col(iam_cols, "activityID")
        c_mat = pick_col(iam_cols, "materialTypeID")
        c_qty = pick_col(iam_cols, "quantity")
        mats_by_bp: Dict[int, List[Tuple[int, int]]] = {}
        for r in conn.execute(
            f"SELECT {c_bp} AS bp, {c_mat} AS mat, {c_qty} AS qty FROM industryActivityMaterials WHERE {c_act}=?;",
            (activity_id,),
        ).fetchall():
            bp = int(r["bp"])
            mats_by_bp.setdefault(bp, []).append((int(r["mat"]), int(r["qty"])))

        # Products
        iap_cols = table_columns(conn, "industryActivityProducts")
        c_bp2 = pick_col(iap_cols, "typeID")
        c_act2 = pick_col(iap_cols, "activityID")
        c_prod = pick_col(iap_cols, "productTypeID")
        c_pqty = pick_col(iap_cols, "quantity")
        recipes: List[Dict[str, Any]] = []
        for r in conn.execute(
            f"SELECT {c_bp2} AS bp, {c_prod} AS prod, {c_pqty} AS pqty FROM industryActivityProducts WHERE {c_act2}=?;",
            (activity_id,),
        ).fetchall():
            bp = int(r["bp"])
            prod = int(r["prod"])
            pqty = int(r["pqty"] or 1)

            bp_info = types.get(bp, {"name": f"type {bp}"})
            prod_info = types.get(prod, {"name": f"type {prod}"})

            # Filter out unpublished products (helps avoid oddities).
            if types.get(prod, {}).get("published", 1) == 0:
                continue

            mats = []
            for mat_id, qty in mats_by_bp.get(bp, []):
                mat_info = types.get(mat_id, {"name": f"type {mat_id}"})
                mats.append({"type_id": mat_id, "name": mat_info["name"], "qty": qty})

            recipes.append(
                {
                    "blueprint_type_id": bp,
                    "blueprint_name": bp_info.get("name", f"type {bp}"),
                    "product_type_id": prod,
                    "product_name": prod_info.get("name", f"type {prod}"),
                    "product_qty": pqty,
                    "time_s": int(activity_time.get((bp, activity_id), 0)),
                    "materials": mats,
                }
            )
        return recipes

    manufacturing = build_industry_recipes(ACTIVITY_MANUFACTURING)
    reactions = build_industry_recipes(ACTIVITY_REACTIONS)

    # Refining / reprocessing outputs (invTypeMaterials)
    itm_cols = table_columns(conn, "invTypeMaterials")
    c_itm_type = pick_col(itm_cols, "typeID")
    c_itm_mat = pick_col(itm_cols, "materialTypeID")
    c_itm_qty = pick_col(itm_cols, "quantity")

    # We’ll capture all published + market-group items that have reprocessing outputs.
    # This includes ores, ice, scrap modules, etc.
    refining_by_input: Dict[int, List[Tuple[int, int]]] = {}
    for r in conn.execute(
        f"SELECT {c_itm_type} AS input_id, {c_itm_mat} AS out_id, {c_itm_qty} AS qty FROM invTypeMaterials;"
    ).fetchall():
        inp = int(r["input_id"])
        out = int(r["out_id"])
        qty = int(r["qty"] or 0)
        if qty <= 0:
            continue
        refining_by_input.setdefault(inp, []).append((out, qty))

    refining: List[Dict[str, Any]] = []
    for inp, outs in refining_by_input.items():
        info = types.get(inp)
        if not info:
            continue
        if info.get("published", 1) == 0:
            continue
        if c_mktgrp and info.get("market_group") is None:
            continue

        outputs = []
        for out_id, qty in outs:
            out_info = types.get(out_id, {"name": f"type {out_id}"})
            outputs.append({"type_id": out_id, "name": out_info["name"], "qty": qty})

        refining.append(
            {
                "input_type_id": inp,
                "input_name": info["name"],
                "portion_size": int(info.get("portion") or 1),
                "volume": float(info.get("volume") or 0.0),
                "outputs": outputs,
            }
        )

    conn.close()

    return {
        "meta": {
            "source": FUZZWORK_SDE_SQLITE_BZ2,
            "note": "Derived recipe data from Fuzzwork's SDE SQLite conversion.",
        },
        "manufacturing": manufacturing,
        "reactions": reactions,
        "refining": refining,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/recipes.json.gz", help="Output gzip JSON file.")
    ap.add_argument("--cache-dir", default=".cache", help="Where to store downloaded SDE DB.")
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists.")
    args = ap.parse_args()

    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"[build_recipes] {out_path} already exists. Use --force to rebuild.")
        return

    bz2_path = cache_dir / "eve.db.bz2"
    db_path = cache_dir / "eve.db"

    if not bz2_path.exists():
        print(f"[build_recipes] Downloading SDE SQLite from {FUZZWORK_SDE_SQLITE_BZ2}")
        download(FUZZWORK_SDE_SQLITE_BZ2, bz2_path)

    if not db_path.exists() or args.force:
        print(f"[build_recipes] Decompressing {bz2_path} -> {db_path}")
        decompress_bz2(bz2_path, db_path)

    print("[build_recipes] Building recipes…")
    recipes = build_recipes(db_path)

    _ensure_parent(out_path)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False)

    print(f"[build_recipes] Wrote {out_path}")


if __name__ == "__main__":
    main()
