from __future__ import annotations

from pathlib import Path


def resolve_data_root(cfg_root: str) -> Path:
    """Return first existing FNSPID root among configured and fallbacks."""
    candidates = [
        Path(cfg_root),
        Path("Data/FNSPID"),
        Path("data/FNSPID"),
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return Path(cfg_root).resolve()


def local_prices_dir(root: Path) -> Path:
    nested = root / "Stock_price" / "full_history" / "full_history"
    if nested.exists():
        return nested
    return root / "Stock_price" / "full_history"


def local_news_path(root: Path, source: str = "nasdaq_exteral_data") -> Path:
    return root / "Stock_news" / f"{source}.csv"
