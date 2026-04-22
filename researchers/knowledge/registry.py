"""
Columnar Knowledge Registry — SAND-inspired bit-vector indexing.
Every knowledge atom is tagged on 5 dimensions. Query by ANY dimension.

    from knowledge.registry import KB
    atoms = KB.query(pairs=KB.BTC, regimes=KB.TRENDING, affects=KB.AFFECTS_SIZING)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import IntFlag

class Pair(IntFlag):
    BTC=32; ETH=16; SOL=8; DOGE=4; AVAX=2; LINK=1; ALL=63; ALTS=31; DEFI=19

class Regime(IntFlag):
    LOW_VOL=8; TRENDING=4; HIGH_VOL=2; TRANSITION=1; ALL=15; STABLE=12; CRISIS=3

class Scale(IntFlag):
    MIN5=16; MIN30=8; HOUR1=4; HOUR4=2; DAY1=1; ALL=31; FAST=24; SLOW=3

class AType(IntFlag):
    FORMULA=32; COMPUTED=16; SIGNAL=8; THRESHOLD=4; REFERENCE=2; QUERY=1; ALL=63

class Affects(IntFlag):
    FEATURES=128; MODELS=64; SIGNALS=32; SIZING=16; RISK=8; EXECUTION=4; REGIME=2; COST=1; ALL=255

@dataclass
class Atom:
    id: str; name: str; domain: str; doc: str
    pairs: int = Pair.ALL; regimes: int = Regime.ALL; scales: int = Scale.ALL
    atype: int = AType.REFERENCE; affects: int = 0
    formula: Optional[Callable] = None; value: Any = None; sql: Optional[str] = None
    crypto_specific: bool = False
    dead_end_ids: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)

class KnowledgeRegistry:
    # Expose flags
    BTC=Pair.BTC; ETH=Pair.ETH; SOL=Pair.SOL; DOGE=Pair.DOGE; AVAX=Pair.AVAX; LINK=Pair.LINK
    ALL_PAIRS=Pair.ALL; ALTS=Pair.ALTS; DEFI=Pair.DEFI
    LOW_VOL=Regime.LOW_VOL; TRENDING=Regime.TRENDING; HIGH_VOL=Regime.HIGH_VOL; TRANSITION=Regime.TRANSITION
    CRISIS=Regime.CRISIS; STABLE=Regime.STABLE
    SCALE_5MIN=Scale.MIN5; SCALE_30MIN=Scale.MIN30; SCALE_1HR=Scale.HOUR1
    FORMULA=AType.FORMULA; COMPUTED=AType.COMPUTED; SIGNAL=AType.SIGNAL; THRESHOLD=AType.THRESHOLD
    AFFECTS_FEATURES=Affects.FEATURES; AFFECTS_MODELS=Affects.MODELS; AFFECTS_SIGNALS=Affects.SIGNALS
    AFFECTS_SIZING=Affects.SIZING; AFFECTS_RISK=Affects.RISK; AFFECTS_EXECUTION=Affects.EXECUTION
    AFFECTS_REGIME=Affects.REGIME; AFFECTS_COST=Affects.COST

    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._by_domain: Dict[str, List[str]] = {}

    def register(self, atom: Atom):
        self._atoms[atom.id] = atom
        self._by_domain.setdefault(atom.domain, []).append(atom.id)

    def register_many(self, atoms: List[Atom]):
        for a in atoms: self.register(a)

    def query(self, pairs=0, regimes=0, scales=0, atype=0, affects=0,
              domain=None, crypto_only=False) -> List[Atom]:
        results = []
        for atom in self._atoms.values():
            if pairs and not (atom.pairs & pairs): continue
            if regimes and not (atom.regimes & regimes): continue
            if scales and not (atom.scales & scales): continue
            if atype and not (atom.atype & atype): continue
            if affects and not (atom.affects & affects): continue
            if domain and atom.domain != domain: continue
            if crypto_only and not atom.crypto_specific: continue
            results.append(atom)
        return results

    def get(self, atom_id: str) -> Optional[Atom]:
        return self._atoms.get(atom_id)

    def execute(self, atom_id: str, **kwargs) -> Any:
        atom = self._atoms.get(atom_id)
        if not atom: raise KeyError(f"Unknown: {atom_id}")
        if not atom.formula: raise TypeError(f"No formula: {atom_id}")
        return atom.formula(**kwargs)

    def explain(self, atom_id: str) -> str:
        atom = self._atoms.get(atom_id)
        if not atom: return f"Unknown: {atom_id}"
        lines = [f"{'='*60}", f"  {atom.name}  [{atom.id}]", f"{'='*60}",
                 f"Domain: {atom.domain}", f"Affects: {Affects(atom.affects).name if atom.affects else 'N/A'}",
                 "", atom.doc]
        if atom.see_also: lines.append(f"\nSee also: {atom.see_also}")
        return "\n".join(lines)

    def manifest(self, domain=None) -> str:
        lines = [f"KNOWLEDGE REGISTRY: {len(self._atoms)} atoms\n"]
        domains = [domain] if domain else sorted(self._by_domain.keys())
        for d in domains:
            ids = self._by_domain.get(d, [])
            if not ids: continue
            lines.append(f"{'─'*50}\n  {d.upper()} ({len(ids)} atoms)\n{'─'*50}")
            for aid in ids:
                atom = self._atoms[aid]
                short = atom.doc.split('\n')[0][:70]
                tag = "⚡" if atom.formula else "📊" if atom.sql else "📖"
                lines.append(f"  {tag} {atom.id:45s} {short}")
        return "\n".join(lines)

    def diagnostic(self, domain: str) -> Dict[str, Any]:
        results = {}
        for atom in self.query(domain=domain):
            if atom.formula:
                try: results[atom.id] = atom.formula()
                except Exception as e: results[atom.id] = {"error": str(e)}
        return results

    def __len__(self): return len(self._atoms)
    def __repr__(self): return f"KnowledgeRegistry({len(self._atoms)} atoms)"

KB = KnowledgeRegistry()
