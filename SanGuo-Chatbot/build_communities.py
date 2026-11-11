"""Build the offline community index with the pinned igraph Leiden implementation."""

from collections import Counter
import json
import random

import igraph

from config import COMMUNITY_INDEX_PATH, COMMUNITY_RANDOM_SEED, QA_FACTS_PATH
from community_search import CommunityIndex
from fact_qa import evidence_fingerprint, load_evidence_facts


def build_index(facts):
    entities = sorted({name for fact in facts for name in (fact.head, fact.tail)})
    indices = {name: index for index, name in enumerate(entities)}
    distinct = {(fact.head, fact.relation, fact.tail, fact.time_id or fact.time_text) for fact in facts}
    weights = Counter(tuple(sorted((indices[h], indices[t]))) for h, _, t, _ in distinct if h != t)
    edges = sorted(weights)
    graph = igraph.Graph(n=len(entities), edges=edges, directed=False)
    igraph.set_random_number_generator(random.Random(COMMUNITY_RANDOM_SEED))
    partition = graph.community_leiden(objective_function="modularity",
                                       weights=[weights[edge] for edge in edges], n_iterations=-1)
    strength = graph.strength(weights=[weights[edge] for edge in edges])
    groups = []
    for ordinal, cluster in enumerate(sorted(sorted(group) for group in partition), 1):
        members = [entities[index] for index in cluster]
        member_set = set(members)
        core = [entities[index] for index in sorted(cluster, key=lambda i: (-strength[i], entities[i]))[:3]]
        groups.append({"id": f"C{ordinal:03d}", "members": members, "core": core,
                       "fact_ids": [fact.fact_id for fact in facts if fact.head in member_set and fact.tail in member_set]})
    CommunityIndex(groups, facts)
    return {"algorithm": "Leiden", "objective": "modularity", "seed": COMMUNITY_RANDOM_SEED,
            "igraph": igraph.__version__, "entities": len(entities), "undirected_edges": len(edges),
            "modularity": partition.modularity, "evidence_sha256": evidence_fingerprint(facts), "communities": groups}


def main():
    facts = load_evidence_facts(QA_FACTS_PATH)
    index = build_index(facts)
    staging = COMMUNITY_INDEX_PATH.with_suffix(".json.tmp")
    staging.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    staging.replace(COMMUNITY_INDEX_PATH)
    print(f"Leiden: {len(index['communities'])} communities, {index['entities']} entities, modularity={index['modularity']:.4f}")


if __name__ == "__main__":
    main()
