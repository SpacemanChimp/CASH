# EVE Money Button — Emperor Edition

This repo builds a static GitHub Pages site that answers one question:

> "What should I build/refine/react today to make ISK in Jita / The Forge?"

## What you get

- **Manufacturing** rankings (T1 / general manufacturing)
- **Reactions** rankings
- **Refining** rankings (ore/ice/moon/compressed inputs)
- **T2 (Invention)** rankings (T1 invention + T2 manufacturing amortized)
- **Confidence score (0–100)** to filter out meme markets
- **Depth-based recommended runs/batches** (orderbook slippage caps) for top candidates
- **Plan Builder** that creates buy/sell lists and estimates slots + hauling

## Local run

```bash
python -m pip install -r requirements.txt
python scripts/update_rankings.py --out docs/data/rankings.json
python -m http.server --directory docs 8000
```

Then open http://localhost:8000

## GitHub Pages

1. Repo Settings → Pages → Source: **GitHub Actions**
2. Run the workflow **Build and Deploy (EVE Money Button)** (or push to `main`)

The workflow refreshes rankings on a schedule (default every 6 hours).
