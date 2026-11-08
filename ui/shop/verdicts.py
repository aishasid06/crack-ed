"""Turn a classified listing into what a shopper needs: a group (buy / be careful / avoid) and one plain
sentence saying why. No scores, rule ids or statutes."""

from __future__ import annotations

from dataclasses import dataclass

from bh.db import Features, Listing
from ui import theme


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    blurb: str
    colour: str
    tone: str  # Streamlit text colour for the collapsible heading (themed in .streamlit/config.toml)


GROUPS = [
    Group("genuine", "Looks genuine", "Official sellers and genuine items being resold.", theme.GREEN, "green"),
    Group("careful", "Likely genuine", "Real items with a catch, such as an import or changes made after sale, "
                                       "so check the details first.", theme.AMBER, "orange"),
    Group("avoid", "Avoid", "Likely fakes, or listings that use the brand's photos to sell something else.",
          theme.RED, "red"),
    Group("lookalike", "Similar products from other makers", "Not the brand you searched for, but similar.",
          theme.MUTED, "gray"),
]
GROUP_BY_KEY = {g.key: g for g in GROUPS}

CATEGORY_GROUP = {
    "authorised": "genuine",
    "genuine_resale": "genuine",
    "grey_import": "careful",
    "modified_goods": "careful",
    "counterfeit_identical": "avoid",
    "counterfeit_similar": "avoid",
    "copied_photos": "avoid",
    "lookalike": "lookalike",
    "unrelated": "hidden",
}

LABEL = {
    "authorised": "Official seller",
    "genuine_resale": "Genuine, resold",
    "grey_import": "Imported",
    "modified_goods": "Modified",
    "counterfeit_identical": "Likely fake",
    "counterfeit_similar": "Copycat name",
    "copied_photos": "Copied photos",
    "lookalike": "Other maker",
}


def _signs(f: Features | None, price_ratio_limit: float) -> list[str]:
    if f is None:
        return []
    out = []
    if f.replica_kw:
        out.append("the seller calls it " + " / ".join(f"“{k}”" for k in f.replica_kw[:2]))
    if f.price_ratio is not None and f.price_ratio < price_ratio_limit:
        out.append(f"it costs {f.price_ratio:.0%} of the official price")
    if f.cluster_suspect and f.cluster_size > 1:
        out.append("the seller has many near-identical listings")
    return out


def reason(category: str, listing: Listing, f: Features | None, brand: str, price_ratio_limit: float = 0.5) -> str:
    signs = _signs(f, price_ratio_limit)
    tail = f": {', and '.join(signs)}." if signs else "."
    if category == "authorised":
        return f"Sold by an official {brand} retailer."
    if category == "genuine_resale":
        return f"Looks like a genuine {brand} item being resold, at a normal price for a real one."
    if category == "grey_import":
        where = f" ({f.origin_evidence})" if f is not None and f.origin_evidence else ""
        return (f"A genuine {brand} item imported from outside the UK{where}. UK warranty and returns "
                f"may not apply.")
    if category == "modified_goods":
        words = f" (“{f.modification_kw[0]}”)" if f is not None and f.modification_kw else ""
        return f"A genuine {brand} item that has been changed since it was sold{words}, so it may not be as new."
    if category == "counterfeit_identical":
        return f"Uses the {brand} name, but shows signs of a fake" + tail
    if category == "counterfeit_similar":
        return f"Uses a name made to look like {brand}" + tail
    if category == "copied_photos":
        return (f"Uses {brand}'s own product photos but doesn't sell it as {brand}. What arrives may not "
                f"match the pictures.")
    if category == "lookalike":
        return f"A similar-looking product from another maker. It isn't {brand}."
    return "Not related to your search."
