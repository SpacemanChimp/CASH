# EVE Money Button (GitHub Pages)

A static site + GitHub Actions pipeline that:

1. Pulls **market aggregates** (Fuzzwork)
2. Compares against **SDE recipes** (manufacturing, reactions, reprocessing)
3. Publishes **top-ranked “do this to make money” picks** to a GitHub Pages site.

> ⚠️ This is decision-support, not a guarantee. Always sanity-check volume, liquidity, hauling risk, industry job fees, and your own bonuses.

---

## What you get

- **Manufacturing**: Top profitable builds (includes blueprint *payback runs* if a BPO market price exists)
- **Reactions**: Top profitable reactions
- **Refining**: Best reprocess/refine candidates **ranked by ISK per m³ of input batch volume** (space efficiency)

The site lives in `/docs` so GitHub Pages can serve it directly.

---

## Setup (10–15 minutes of clicking)

1. Create a new GitHub repo (e.g. `eve-money-button`)
2. Copy the contents of this project into it
3. In GitHub:
   - **Settings → Pages**
   - Source: `Deploy from a branch`
   - Branch: `main` / Folder: `/docs`
4. In GitHub:
   - **Settings → Actions → General**
   - Workflow permissions: **Read and write**
5. Go to **Actions → “Update EVE Money Button Rankings” → Run workflow**
   - First run will also build `data/recipes.json.gz` by downloading the SDE SQLite.

After the workflow commits `docs/data/rankings.json`, your site will show the tables.

---

## Customizing assumptions

In `.github/workflows/update.yml`, edit the script flags:

- `--price-mode`:
  - `minmax` = buy materials at min sell, sell outputs at max buy (fast execution)
  - `percentile` = 5% average (smoother / less outlier sensitivity)
  - `weighted` = weighted average
- `--fee-rate` = tax/fees to subtract from revenue (default: 0.03)
- `--reprocess-yield` = your effective refine yield (default: 0.7)

Example:

```bash
python scripts/update_rankings.py --price-mode minmax --fee-rate 0.015 --reprocess-yield 0.82
```

---

## Updating recipe data (SDE changes)

Recipe data changes rarely, but when it does:

- Run the workflow with `--force-recipes`
  - or delete `data/recipes.json.gz` and run again.

---

## Roadmap ideas (easy next wins)

- Add filters: “only T1”, “exclude capitals”, “min daily volume”
- Add ESI login (SSO) to personalize:
  - owned blueprints
  - skills / structure bonuses
  - current inventory/assets
- Add “shopping list” and “sell list” export formats

---

## License

MIT
