"""Pre-cache per-ticker news parquet files for faster dataset builds."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from src.data.dataset import TickerDataStore
from src.data.paths import resolve_data_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache FNSPID news per ticker.")
    parser.add_argument(
        "--config",
        default="configs/data/fnspid.yaml",
        help="Hydra data config (tickers.dev used by default).",
    )
    parser.add_argument("--ticker-set", default="dev", choices=["dev", "paper"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = resolve_data_root(cfg.root)
    store = TickerDataStore(root, cfg.news_source, list(cfg.features), cfg.target)
    tickers = list(cfg.tickers[args.ticker_set])
    for ticker in tickers:
        news = store.get_news(ticker)
        print(f"{ticker}: {len(news)} news rows cached")


if __name__ == "__main__":
    main()
