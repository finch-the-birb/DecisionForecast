"""Pre-cache per-ticker news parquet files for faster dataset builds."""

from __future__ import annotations

import argparse
import logging

from omegaconf import OmegaConf

from src.data.dataset import precache_news_for_tickers
from src.data.paths import resolve_data_root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Cache FNSPID news per ticker.")
    parser.add_argument(
        "--config",
        default="configs/data/fnspid.yaml",
        help="Hydra data config (tickers.dev used by default).",
    )
    parser.add_argument("--ticker-set", default="dev", choices=["dev", "paper"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = resolve_data_root(str(cfg.root))
    tickers = list(cfg.tickers[args.ticker_set])
    counts = precache_news_for_tickers(root, str(cfg.news_source), tickers)
    for ticker in tickers:
        print(f"{ticker}: {counts.get(ticker, 0)} news rows cached")


if __name__ == "__main__":
    main()
