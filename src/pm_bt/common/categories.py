from __future__ import annotations

import polars as pl

# Ordered from more specific families to broad catch-alls.
_KALSHI_GROUP_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("NCAAF", "NCAAMB", "NFL", "NBA", "MLB", "NHL", "WNBA", "UFC", "F1", "PGA"), "sports"),
    (
        (
            "PRES",
            "SENATE",
            "HOUSE",
            "GOV",
            "VOTE",
            "TRUMP",
            "BIDEN",
            "EC",
            "MAYOR",
            "ELECTION",
            "CABINET",
        ),
        "politics",
    ),
    (("BTC", "ETH", "DOGE", "SOL", "XRP"), "crypto"),
    (
        (
            "FED",
            "CPI",
            "GDP",
            "NASDAQ",
            "INX",
            "TNOTE",
            "USDJPY",
            "EURUSD",
            "WTI",
            "TARIFF",
        ),
        "finance",
    ),
    (("RAIN", "SNOW", "HIGH", "HUR", "TORNADO", "WEATHER"), "weather"),
    (
        (
            "OSCAR",
            "GRAMMY",
            "EMMY",
            "SPOTIFY",
            "NETFLIX",
            "BILLBOARD",
            "TOPSONG",
            "TOPARTIST",
            "RT",
        ),
        "entertainment",
    ),
    (("LLM", "AI", "SPACEX", "APPLE"), "science_tech"),
    (("MENTION", "HEADLINE", "GOOGLESEARCH"), "media"),
)


def classify_kalshi_event_ticker(event_ticker: str | None) -> str | None:
    """Classify a Kalshi ``event_ticker`` into a normalized top-level category."""
    if event_ticker is None:
        return None
    ticker_upper = event_ticker.upper()
    ticker_code = ticker_upper.split("-", 1)[0]
    for patterns, group in _KALSHI_GROUP_PATTERNS:
        if any(ticker_code.startswith(pattern) for pattern in patterns):
            return group
    return "other"


def kalshi_category_expr(column: str = "event_ticker") -> pl.Expr:
    """Vectorized Polars expression version of ``classify_kalshi_event_ticker``."""
    ticker = pl.col(column).cast(pl.Utf8, strict=False).fill_null("").str.to_uppercase()
    ticker_code = ticker.str.extract(r"^([A-Z0-9]+)", 1).fill_null("")

    expr: pl.Expr = pl.lit("other")
    for patterns, group in reversed(_KALSHI_GROUP_PATTERNS):
        matches = pl.any_horizontal([ticker_code.str.starts_with(pat) for pat in patterns])
        expr = pl.when(matches).then(pl.lit(group)).otherwise(expr)

    return pl.when(pl.col(column).is_null()).then(pl.lit(None, dtype=pl.Utf8)).otherwise(expr)


__all__ = [
    "classify_kalshi_event_ticker",
    "kalshi_category_expr",
]
