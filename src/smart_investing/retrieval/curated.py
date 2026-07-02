"""Curated supply-chain seed edges.

Co-mention edges (from filing text) are broad but noisy and directional; big
filers often don't name peers by their distinctive bigram, so isolated nodes
appear. This hand-curated map encodes the *canonical* relationships for the
sectors in our corpus so the supply-chain walk is reliable and legible:

  - SUPPLIES[supplier] = [customers...]   (supplier provides inputs downstream)
  - COMPETES = [cluster, ...]             (rivals at the same layer)

`curated_edges(known)` expands these to (src, dst, rel, weight) tuples, keeping
only tickers actually present in the corpus. Everything is tradeable (all are
US-listed filers we ingested). Edge direction is supplier -> customer.
"""

from __future__ import annotations

from itertools import combinations

# supplier -> the customers it feeds (upstream -> downstream)
SUPPLIES: dict[str, list[str]] = {
    # --- semiconductor equipment & materials -> chipmakers/fabs ---
    "AMAT": ["INTC", "MU", "TXN", "ON", "MCHP"],
    "LRCX": ["INTC", "MU", "TXN", "ON"],
    "KLAC": ["INTC", "MU", "TXN"],
    "TER": ["INTC", "MU", "NVDA", "AMD"],
    "ONTO": ["INTC", "MU", "AMAT"],
    "ACLS": ["ON", "INTC", "MU"],
    "FORM": ["INTC", "MU", "TER"],
    "MKSI": ["AMAT", "LRCX", "INTC"],
    "AEIS": ["AMAT", "LRCX", "KLAC"],
    "COHR": ["AMAT", "NVDA", "CIEN"],
    "ENTG": ["INTC", "MU", "AMAT", "LRCX"],
    # --- IP / fabless chip designers -> downstream systems & clouds ---
    "ARM": ["NVDA", "AMD", "QCOM", "AVGO", "AMZN"],
    "NVDA": ["SMCI", "DELL", "HPE", "ANET", "MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "AMD": ["SMCI", "DELL", "HPE", "MSFT", "GOOGL", "META", "AMZN"],
    "INTC": ["DELL", "HPE", "SMCI"],
    "MU": ["NVDA", "DELL", "SMCI", "HPE"],
    "AVGO": ["ANET", "CIEN", "DELL", "GOOGL", "META"],
    "MRVL": ["ANET", "CIEN", "AMZN", "MSFT"],
    "QCOM": ["DELL", "META"],
    "MPWR": ["NVDA", "DELL", "SMCI"],
    "SWKS": ["AVGO"],
    "QRVO": ["AVGO"],
    "ADI": ["INTC", "DELL"],
    "TXN": ["DELL", "HPE"],
    "ON": ["DELL", "VRT"],
    # --- servers / networking / storage -> hyperscalers & enterprise ---
    "SMCI": ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "DELL": ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "HPE": ["MSFT", "GOOGL", "AMZN"],
    "ANET": ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "CIEN": ["MSFT", "GOOGL", "AMZN"],
    "NTAP": ["MSFT", "AMZN", "ORCL"],
    "WDC": ["DELL", "HPE", "NTAP", "AMZN"],
    "STX": ["DELL", "HPE", "NTAP", "GOOGL"],
    # --- datacenter power & cooling -> datacenters / hyperscalers ---
    "VRT": ["SMCI", "DELL", "HPE", "MSFT", "GOOGL", "META", "AMZN"],
    "ETN": ["VRT", "SMCI", "MSFT", "GOOGL", "AMZN"],
    "GEV": ["CEG", "VST", "NEE"],
    "PWR": ["CEG", "VST", "NEE", "NRG"],
    "CW": ["LMT", "NOC", "BWXT"],
    # --- power generation -> hyperscaler PPAs (AI datacenter demand) ---
    "CEG": ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "VST": ["MSFT", "GOOGL", "AMZN"],
    "NRG": ["MSFT", "META"],
    "TLN": ["AMZN", "GOOGL"],
    "NEE": ["MSFT", "GOOGL", "AMZN"],
    "FSLR": ["NEE", "META", "AMZN"],
    "ENPH": ["NEE"],
    # --- nuclear fuel chain: miners -> enrichment/fuel -> reactors/utilities ---
    "CCJ": ["LEU", "CEG", "VST", "NEE"],
    "UEC": ["LEU", "CEG"],
    "UUUU": ["LEU"],
    "DNN": ["LEU"],
    "LEU": ["SMR", "OKLO", "NNE", "CEG", "VST"],
    "BWXT": ["SMR", "OKLO", "NNE", "NOC", "CEG"],
    "SMR": ["CEG", "VST", "NEE"],
    "OKLO": ["CEG", "VST"],
    "NNE": ["CEG"],
}

# rivals at the same layer (undirected within each cluster)
COMPETES: list[list[str]] = [
    ["NVDA", "AMD", "INTC"],
    ["AMAT", "LRCX", "KLAC", "TER", "ONTO", "ACLS"],
    ["TXN", "ADI", "MCHP", "ON", "MPWR"],
    ["SWKS", "QRVO", "AVGO"],
    ["MSFT", "GOOGL", "AMZN", "ORCL"],
    ["META", "GOOGL", "MSFT"],
    ["SMCI", "DELL", "HPE"],
    ["ANET", "CIEN"],
    ["WDC", "STX", "NTAP"],
    ["CEG", "VST", "NRG", "TLN"],
    ["CCJ", "UEC", "UUUU", "DNN"],
    ["SMR", "OKLO", "NNE"],
    ["LMT", "NOC", "RTX", "LHX"],
    ["IONQ", "RGTI", "QBTS"],
    ["FSLR", "ENPH"],
]


def curated_edges(known: set[str]) -> list[tuple[str, str, str, float]]:
    """Expand the seed map to (src, dst, rel, weight), keeping only known tickers."""
    edges: list[tuple[str, str, str, float]] = []
    for supplier, customers in SUPPLIES.items():
        if supplier not in known:
            continue
        for cust in customers:
            if cust in known and cust != supplier:
                edges.append((supplier, cust, "supplies", 1.0))
    for cluster in COMPETES:
        members = [m for m in cluster if m in known]
        for a, b in combinations(members, 2):
            edges.append((a, b, "competes", 0.6))
    return edges
