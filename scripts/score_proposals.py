"""
Score proposals by cross-specialist peer review consensus.
Reads proposals and reviews from a council session directory.
Outputs a ranked list that feeds into the existing SafetyGate pipeline.

Usage: .venv/bin/python3 scripts/score_proposals.py data/research_sessions/latest/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

RESEARCHERS = ['mathematician', 'cryptographer', 'physicist', 'linguist', 'systems_engineer']

# Domain groupings for cross-domain bonus
DOMAIN_GROUPS = {
    'mathematician': 'theory',
    'cryptographer': 'signals',
    'physicist': 'data',
    'linguist': 'models',
    'systems_engineer': 'infrastructure',
}


def load_all_proposals(session_dir: Path) -> List[Dict]:
    """Load proposals from all researchers, tagging each with its source."""
    all_proposals = []
    for name in RESEARCHERS:
        prop_file = session_dir / "proposals" / name / "proposals.json"
        if not prop_file.exists():
            continue
        try:
            proposals = json.loads(prop_file.read_text())
            for p in proposals:
                p["_source_researcher"] = name
            all_proposals.extend(proposals)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: could not load {prop_file}: {e}")
    return all_proposals


def load_all_reviews(session_dir: Path) -> List[Dict]:
    """Load reviews from all researchers."""
    all_reviews = []
    for name in RESEARCHERS:
        review_file = session_dir / "reviews" / name / "reviews.json"
        if not review_file.exists():
            continue
        try:
            reviews = json.loads(review_file.read_text())
            for r in reviews:
                r["_reviewer"] = name
            all_reviews.extend(reviews)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: could not load {review_file}: {e}")
    return all_reviews


def score(session_dir: Path) -> List[Dict]:
    """
    Score each proposal based on peer reviews.

    Scoring:
      endorse:   +2.0 x confidence
      challenge: +0.5 x confidence
      reject:    -1.0 x confidence
      Cross-domain endorsement bonus: +1.0 x confidence

    Consensus threshold: >=2 endorsements and rejections < endorsements.
    """
    proposals = load_all_proposals(session_dir)
    reviews = load_all_reviews(session_dir)

    sources = len(set(p['_source_researcher'] for p in proposals)) if proposals else 0
    reviewers = len(set(r['_reviewer'] for r in reviews)) if reviews else 0
    print(f"Loaded {len(proposals)} proposals from {sources} researchers")
    print(f"Loaded {len(reviews)} reviews from {reviewers} reviewers")

    for proposal in proposals:
        source = proposal["_source_researcher"]
        title = proposal.get("title", "untitled")
        score_val = 0.0
        endorsements = 0
        rejections = 0

        # Find reviews for this proposal (match by title and source researcher)
        matching_reviews = [
            r for r in reviews
            if r.get("researcher") == source and r.get("proposal_title") == title
        ]

        for review in matching_reviews:
            reviewer = review["_reviewer"]
            confidence = review.get("confidence", 0.5)
            verdict = review.get("verdict", "").lower()

            if verdict == "endorse":
                score_val += 2.0 * confidence
                endorsements += 1
                # Cross-domain bonus
                if DOMAIN_GROUPS.get(reviewer) != DOMAIN_GROUPS.get(source):
                    score_val += 1.0 * confidence

            elif verdict == "challenge":
                score_val += 0.5 * confidence

            elif verdict == "reject":
                score_val -= 1.0 * confidence
                rejections += 1

        proposal["consensus_score"] = round(score_val, 2)
        proposal["endorsements"] = endorsements
        proposal["rejections"] = rejections
        proposal["passes_consensus"] = endorsements >= 2 and rejections < endorsements
        proposal["review_count"] = len(matching_reviews)

    # Sort by consensus score descending
    ranked = sorted(proposals, key=lambda p: p["consensus_score"], reverse=True)

    # Print summary
    print(f"\nRanked proposals:")
    for i, p in enumerate(ranked):
        icon = "PASS" if p["passes_consensus"] else "FAIL"
        print(f"  {i+1}. [{icon}] [{p['_source_researcher']}] {p.get('title', 'untitled')} "
              f"score={p['consensus_score']} endorse={p['endorsements']} reject={p['rejections']}")

    # Save ranked results
    output_path = session_dir / "ranked_proposals.json"
    output_path.write_text(json.dumps(ranked, indent=2, default=str))
    print(f"\nRanked proposals saved to {output_path}")

    # Return only proposals passing consensus (for SafetyGate)
    passing = [p for p in ranked if p["passes_consensus"]]
    print(f"{len(passing)}/{len(ranked)} proposals pass consensus threshold")
    return passing


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/score_proposals.py <session_directory>")
        sys.exit(1)
    score(Path(sys.argv[1]))
