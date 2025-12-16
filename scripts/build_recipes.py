#!/usr/bin/env python3
"""
build_recipes.py

Builds a compact recipe cache (data/recipes.json.gz) from Fuzzwork's eve.db
(SQLite version of the EVE SDE).

Outputs:
- manufacturing recipes (activityID=1)
- reaction recipes (activityID=11)
- refining recipes (from invTypeMaterials for ore/ice-ish items)
- invention mapping (activityID=8): T1 blueprint -> T2 blueprint + materials + probability + expected runs

Notes:
- This intentionally avoids heavy SDE joins at runtime by precomputing only what we need.
- Fuzzwork's eve.db schema occasionally changes; this script auto-detects column names.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


EVE_DB_URL = "https://www.fuzzwork.co.uk/dump/latest/eve.db.bz2"
CACHE_DIR = Path(".cache")
EVE_DB_BZ2 = CACHE_DIR / "eve.db.bz2"
EVE_DB_PATH = CACHE_DIR / "eve.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, timeout: int = 120) -> None:
    ensure_parent(dest)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_eve_db(force: bool = False) -> None:
    ensure_parent(EVE_DB_PATH)
    if force or not EVE_DB_PATH.exists():
        if force or not EVE_DB_BZ2.exists():
            print(f"[build_recipes] Downloading {EVE_DB_URL}")
            download(EVE_DB_URL, EVE_DB_BZ2)

        print(f"[build_recipes] Decompressing eve.db.bz2 → eve.db")
        with bz2.open(EVE_DB_BZ2, "rb") as src, open(EVE_DB_PATH, "wb") as dst:
            dst.write(src.read())


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def pick_col(conn: sqlite3.Connection, table: str, candidates: Sequence[str]) -> str:
    cols = set(table_columns(conn, table))
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"No matching column in {table}. Candidates={candidates} available={sorted(cols)[:50]}...")


def try_pick_col(conn: sqlite3.Connection, table: str, candidates: Sequence[str]) -> Optional[str]:
    try:
        return pick_col(conn, table, candidates)
    except Exception:
        return None


def load_types(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    t_typeid = pick_col(conn, "invTypes", ["typeID"])
    t_typename = pick_col(conn, "invTypes", ["typeName", "name"])
    t_volume = try_pick_col(conn, "invTypes", ["volume"])
    t_published = try_pick_col(conn, "invTypes", ["published"])

    stmt = f"SELECT {t_typeid} AS typeID, {t_typename} AS typeName"
    if t_volume:
        stmt += f", COALESCE({t_volume}, 0) AS volume"
    else:
        stmt += ", 0 AS volume"
    if t_published:
        stmt += f", COALESCE({t_published}, 1) AS published"
    else:
        stmt += ", 1 AS published"
    stmt += " FROM invTypes"

    out: Dict[int, Dict[str, Any]] = {}
    for row in conn.execute(stmt).fetchall():
        if int(row["published"]) != 1:
            continue
        tid = int(row["typeID"])
        out[tid] = {
            "name": row["typeName"],
            "volume": float(row["volume"] or 0.0),
            "published": True,
        }
    return out


def load_blueprint_limits(conn: sqlite3.Connection) -> Dict[int, int]:
    """
    Returns maxProductionLimit for blueprint types, when available.
    Not all SDE sqlite variants include this.
    """
    tables = set(list_tables(conn))
    out: Dict[int, int] = {}

    # Classic table name.
    if "invBlueprintTypes" in tables:
        bp_col = try_pick_col(conn, "invBlueprintTypes", ["blueprintTypeID", "typeID"])
        max_col = try_pick_col(conn, "invBlueprintTypes", ["maxProductionLimit", "maxRuns", "productionLimit"])
        if bp_col and max_col:
            for row in conn.execute(f"SELECT {bp_col} AS bp, {max_col} AS maxRuns FROM invBlueprintTypes").fetchall():
                try:
                    bp = int(row["bp"])
                    mr = int(row["maxRuns"]) if row["maxRuns"] is not None else 0
                    if mr > 0:
                        out[bp] = mr
                except Exception:
                    continue

    # Some sqlite exports have industryBlueprints.
    if not out and "industryBlueprints" in tables:
        # Try common column layouts.
        bp_col = try_pick_col(conn, "industryBlueprints", ["blueprintTypeID", "typeID"])
        max_col = try_pick_col(conn, "industryBlueprints", ["maxProductionLimit", "maxRuns", "productionLimit"])
        if bp_col and max_col:
            for row in conn.execute(f"SELECT {bp_col} AS bp, {max_col} AS maxRuns FROM industryBlueprints").fetchall():
                try:
                    bp = int(row["bp"])
                    mr = int(row["maxRuns"]) if row["maxRuns"] is not None else 0
                    if mr > 0:
                        out[bp] = mr
                except Exception:
                    continue

    return out


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

    prod_stmt = (
        f"SELECT a.{ia_bp} AS bp, a.{ia_time} AS t, p.{iap_prod} AS prod, p.{iap_qty} AS qty "
        f"FROM industryActivity a "
        f"JOIN industryActivityProducts p ON a.{ia_bp} = p.{iap_bp} AND a.{ia_act} = p.{iap_act} "
        f"WHERE a.{ia_act} = ?"
    )

    best_product: Dict[int, Tuple[int, float]] = {}
    time_s: Dict[int, int] = {}

    for row in conn.execute(prod_stmt, (activity_id,)).fetchall():
        bp = int(row["bp"])
        t = int(row["t"]) if row["t"] is not None else 0
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
                "activity_id": activity_id,
                "blueprint_type_id": bp,
                "blueprint_name": bp_name,
                "product_type_id": prod,
                "product_name": prod_name,
                "output_qty": out_qty,
                "time_s": int(time_s.get(bp, 0)),
                "materials": mats,
            }
        )

    return out


def build_invention_recipes(
    conn: sqlite3.Connection,
    types: Dict[int, Dict[str, Any]],
    blueprint_limits: Dict[int, int],
) -> List[Dict[str, Any]]:
    """
    Builds invention mapping from industryActivity* where activityID=8 (invention).

    Output entries map a T1 blueprint to the T2 blueprint it can invent, along with:
    - base probability (if available in this eve.db variant)
    - invention time (seconds)
    - materials (datacores, decryptors if listed, etc.)
    - expected runs on the resulting T2 BPC (maxProductionLimit of the T2 blueprint when available; fallback 1)
    """
    ACT_INVENTION = 8

    ia_bp = pick_col(conn, "industryActivity", ["blueprintTypeID", "typeID"])
    ia_act = pick_col(conn, "industryActivity", ["activityID"])
    ia_time = pick_col(conn, "industryActivity", ["time", "duration"])
    ia_prob_col = try_pick_col(conn, "industryActivity", ["probability", "successProbability"])

    iap_bp = pick_col(conn, "industryActivityProducts", ["blueprintTypeID", "typeID"])
    iap_act = pick_col(conn, "industryActivityProducts", ["activityID"])
    iap_prod = pick_col(conn, "industryActivityProducts", ["productTypeID"])
    iap_qty = pick_col(conn, "industryActivityProducts", ["quantity", "qty"])

    iam_bp = pick_col(conn, "industryActivityMaterials", ["blueprintTypeID", "typeID"])
    iam_act = pick_col(conn, "industryActivityMaterials", ["activityID"])
    iam_mat = pick_col(conn, "industryActivityMaterials", ["materialTypeID"])
    iam_qty = pick_col(conn, "industryActivityMaterials", ["quantity", "qty"])

    # Probability table varies. Try to locate one.
    prob_table: Optional[str] = None
    prob_bp_col: Optional[str] = None
    prob_act_col: Optional[str] = None
    prob_p_col: Optional[str] = None

    for tname in list_tables(conn):
        low = tname.lower()
        if low.startswith("industryactivity") and "prob" in low:
            # Candidate table name.
            cols = set(table_columns(conn, tname))
            bp = next((c for c in ["blueprintTypeID", "typeID"] if c in cols), None)
            act = next((c for c in ["activityID"] if c in cols), None)
            p = next((c for c in ["probability", "successProbability", "chance"] if c in cols), None)
            if bp and act and p:
                prob_table, prob_bp_col, prob_act_col, prob_p_col = tname, bp, act, p
                break

    prob_map: Dict[int, float] = {}
    if prob_table and prob_bp_col and prob_act_col and prob_p_col:
        try:
            for row in conn.execute(
                f"SELECT {prob_bp_col} AS bp, {prob_p_col} AS p FROM {prob_table} WHERE {prob_act_col} = ?",
                (ACT_INVENTION,),
            ).fetchall():
                try:
                    bp = int(row["bp"])
                    p = float(row["p"]) if row["p"] is not None else 0.0
                    if p > 0:
                        prob_map[bp] = p
                except Exception:
                    continue
        except Exception:
            prob_map = {}

    prod_stmt = (
        f"SELECT a.{ia_bp} AS bp, a.{ia_time} AS t, "
        f"       p.{iap_prod} AS prod, p.{iap_qty} AS qty"
    )
    if ia_prob_col:
        prod_stmt += f", a.{ia_prob_col} AS prob"
    else:
        prod_stmt += ", NULL AS prob"
    prod_stmt += (
        " FROM industryActivity a "
        " JOIN industryActivityProducts p ON a.{ia_bp} = p.{iap_bp} AND a.{ia_act} = p.{iap_act} "
        " WHERE a.{ia_act} = ?"
    ).format(ia_bp=ia_bp, iap_bp=iap_bp, ia_act=ia_act, iap_act=iap_act)

    # For invention, productTypeID is typically the *T2 blueprint type*.
    best_output: Dict[int, Tuple[int, float]] = {}
    time_s: Dict[int, int] = {}
    base_prob: Dict[int, float] = {}

    for row in conn.execute(prod_stmt, (ACT_INVENTION,)).fetchall():
        bp = int(row["bp"])
        t = int(row["t"]) if row["t"] is not None else 0
        prod = int(row["prod"])
        qty = float(row["qty"]) if row["qty"] is not None else 1.0

        time_s[bp] = t
        if row["prob"] is not None:
            try:
                p = float(row["prob"])
                if p > 0:
                    base_prob[bp] = p
            except Exception:
                pass
        if bp in prob_map:
            base_prob[bp] = prob_map[bp]

        prev = best_output.get(bp)
        if prev is None or qty > prev[1]:
            best_output[bp] = (prod, qty)

    mats_stmt = (
        f"SELECT {iam_mat} AS mat, {iam_qty} AS qty "
        f"FROM industryActivityMaterials WHERE {iam_bp} = ? AND {iam_act} = ?"
    )

    out: List[Dict[str, Any]] = []
    for t1_bp, (t2_bp, _qty) in best_output.items():
        mats: List[Dict[str, Any]] = []
        for mrow in conn.execute(mats_stmt, (t1_bp, ACT_INVENTION)).fetchall():
            mt = int(mrow["mat"])
            q = float(mrow["qty"]) if mrow["qty"] is not None else 0.0
            if q <= 0:
                continue
            mats.append({"type_id": mt, "name": types.get(mt, {}).get("name", f"type {mt}"), "qty": q})

        # If no invention materials are listed, skip (probably not a real invention path).
        if not mats:
            continue

        p = float(base_prob.get(t1_bp, 0.0))
        if p <= 0:
            # Common default base chance; real chance depends on skills/decryptors etc.
            # We store base to allow later scaling; this keeps tool usable when SDE omits probability data.
            p = 0.4

        runs = int(blueprint_limits.get(t2_bp, 0) or 0)
        if runs <= 0:
            runs = 1

        out.append(
            {
                "t1_blueprint_type_id": t1_bp,
                "t1_blueprint_name": types.get(t1_bp, {}).get("name", f"type {t1_bp}"),
                "t2_blueprint_type_id": t2_bp,
                "t2_blueprint_name": types.get(t2_bp, {}).get("name", f"type {t2_bp}"),
                "probability": p,
                "time_s": int(time_s.get(t1_bp, 0)),
                "runs_per_success": runs,
                "materials": mats,
            }
        )

    return out


def build_refining_recipes(conn: sqlite3.Connection, types: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds reprocessing recipes from invTypeMaterials, restricted to ore/ice inputs.

    Important: invTypeMaterials exists for many reprocessable items (modules, loot).
    We *do not* want "refine everything" by default; we want ores/ice/moon ores,
    compressed variants, etc. We identify those using category/group names.
    """
    t_typeid = pick_col(conn, "invTypes", ["typeID"])
    t_typename = pick_col(conn, "invTypes", ["typeName", "name"])
    t_volume = try_pick_col(conn, "invTypes", ["volume"])
    t_portion = try_pick_col(conn, "invTypes", ["portionSize", "portion"])
    t_groupid = pick_col(conn, "invTypes", ["groupID"])
    t_published = try_pick_col(conn, "invTypes", ["published"])

    g_groupid = pick_col(conn, "invGroups", ["groupID"])
    g_groupname = pick_col(conn, "invGroups", ["groupName", "name"])
    g_catid = pick_col(conn, "invGroups", ["categoryID"])

    c_catid = pick_col(conn, "invCategories", ["categoryID"])
    c_catname = pick_col(conn, "invCategories", ["categoryName", "name"])

    itm_typeid = pick_col(conn, "invTypeMaterials", ["typeID"])
    itm_mat = pick_col(conn, "invTypeMaterials", ["materialTypeID"])
    itm_qty = pick_col(conn, "invTypeMaterials", ["quantity", "qty"])

    stmt = (
        f"SELECT t.{t_typeid} AS typeID, t.{t_typename} AS typeName, "
        f"       COALESCE(t.{t_volume}, 0) AS volume, COALESCE(t.{t_portion}, 1) AS portionSize, "
        f"       COALESCE(g.{g_groupname}, '') AS groupName, COALESCE(c.{c_catname}, '') AS categoryName "
        f"FROM invTypes t "
        f"JOIN invGroups g ON t.{t_groupid} = g.{g_groupid} "
        f"JOIN invCategories c ON g.{g_catid} = c.{c_catid} "
        f"WHERE COALESCE(t.{t_published}, 1) = 1 "
        f"  AND t.{t_typeid} IN (SELECT DISTINCT {itm_typeid} FROM invTypeMaterials)"
    )

    def looks_like_refine_input(group_name: str, cat_name: str, type_name: str) -> bool:
        g = (group_name or "").lower()
        c = (cat_name or "").lower()
        t = (type_name or "").lower()

        # Strong signal: category contains asteroid/ice/moon.
        if "asteroid" in c or "ice" in c or "moon" in c:
            return True

        # Some dumps use category names that don't include "asteroid" for ice.
        if "ice" in g:
            return True

        # Compressed variants often show up in group or type names.
        if "compressed" in g or t.startswith("compressed "):
            # Require it to still look like a raw resource.
            if "ore" in c or "asteroid" in c or "ice" in c or "moon" in c:
                return True
            # If we don't have good category names, still allow.
            return True

        return False

    candidates: List[Tuple[int, str, float, int, str, str]] = []
    for row in conn.execute(stmt).fetchall():
        tid = int(row["typeID"])
        tname = str(row["typeName"])
        vol = float(row["volume"] or 0.0)
        portion = int(float(row["portionSize"] or 1))
        gname = str(row["groupName"] or "")
        cname = str(row["categoryName"] or "")

        if not looks_like_refine_input(gname, cname, tname):
            continue
        if portion <= 0:
            portion = 1

        candidates.append((tid, tname, vol, portion, gname, cname))

    mats_stmt = f"SELECT {itm_mat} AS mat, {itm_qty} AS qty FROM invTypeMaterials WHERE {itm_typeid} = ?"

    out: List[Dict[str, Any]] = []
    for tid, tname, vol, portion, gname, cname in candidates:
        outs: List[Dict[str, Any]] = []
        for mrow in conn.execute(mats_stmt, (tid,)).fetchall():
            mt = int(mrow["mat"])
            q = float(mrow["qty"]) if mrow["qty"] is not None else 0.0
            if q <= 0:
                continue
            outs.append({"type_id": mt, "name": types.get(mt, {}).get("name", f"type {mt}"), "qty": q})

        if not outs:
            continue

        batch_units = int(portion)
        v = float(vol or 0.0)
        batch_m3 = float(batch_units) * v

        out.append(
            {
                "input_type_id": tid,
                "input_name": tname,
                "batch_units": batch_units,
                "batch_m3": batch_m3,
                "outputs": outs,
                "meta": {"group": gname, "category": cname},
            }
        )

    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/recipes.json.gz")
    ap.add_argument("--force", action="store_true", help="Re-download and rebuild cache")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    ensure_parent(out_path)

    ensure_eve_db(force=args.force)

    print(f"[build_recipes] Opening SQLite: {EVE_DB_PATH}")
    conn = sqlite3.connect(str(EVE_DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        types = load_types(conn)
        print(f"[build_recipes] Loaded {len(types)} published types")

        blueprint_limits = load_blueprint_limits(conn)
        if blueprint_limits:
            print(f"[build_recipes] Blueprint maxProductionLimit loaded for {len(blueprint_limits)} blueprints")
        else:
            print(f"[build_recipes] Blueprint maxProductionLimit not available in this eve.db; invention run counts will default to 1")

        manufacturing = build_industry_recipes(conn, types, activity_id=1)
        print(f"[build_recipes] Manufacturing recipes: {len(manufacturing)}")

        reactions = build_industry_recipes(conn, types, activity_id=11)
        print(f"[build_recipes] Reaction recipes: {len(reactions)}")

        refining = build_refining_recipes(conn, types)
        print(f"[build_recipes] Refining recipes: {len(refining)}")

        invention = build_invention_recipes(conn, types, blueprint_limits)
        print(f"[build_recipes] Invention mappings: {len(invention)}")

        payload = {
            "generated_at": utc_now_iso(),
            "types": {str(k): v for k, v in types.items()},
            "manufacturing": manufacturing,
            "reactions": reactions,
            "refining": refining,
            "invention": invention,
        }

        print(f"[build_recipes] Writing: {out_path}")
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
