from src.data.schema import RELATION_VOCAB
import numpy as np


def validate_plan(plan: dict, num_shots: int) -> tuple[bool, str]:
    if "entities" not in plan or "shots" not in plan:
        return False, "missing entities or shots"
    if len(plan["shots"]) != num_shots:
        return False, f"expected {num_shots} shots, got {len(plan['shots'])}"
    entities = set(plan["entities"])
    for shot in plan["shots"]:
        for e in shot.get("active_entities", []):
            if e not in entities:
                return False, f"entity '{e}' not in global entities"
        for rel in shot.get("relations", []):
            if len(rel) != 3:
                return False, "relation must have 3 elements"
            subj, rel_type, obj = rel
            if subj not in entities or obj not in entities:
                return False, f"relation entity not in global entities"
            if rel_type not in RELATION_VOCAB:
                rel[1] = "none"
    return True, "ok"


def build_presence_matrix(plan: dict) -> np.ndarray:
    entities = plan["entities"]
    ent2idx = {e: i for i, e in enumerate(entities)}
    S = len(plan["shots"])
    E = len(entities)
    P = np.zeros((S, E), dtype=np.int64)
    for s, shot in enumerate(plan["shots"]):
        for e in shot.get("active_entities", []):
            if e in ent2idx:
                P[s, ent2idx[e]] = 1
    return P


def compute_states(P: np.ndarray, entities: list[str]) -> list[list[str]]:
    S, E = P.shape
    states = [["absent"] * E for _ in range(S)]
    for e in range(E):
        for s in range(S):
            prev = P[s - 1, e] == 1 if s > 0 else False
            cur = P[s, e] == 1
            if not cur:
                states[s][e] = "absent"
            elif not prev and s == 0:
                states[s][e] = "initial"
            elif not prev:
                # check if entity was ever present before
                ever_before = any(P[ss, e] == 1 for ss in range(s))
                states[s][e] = "re_entry" if ever_before else "entry"
            elif prev and cur:
                next_present = P[s + 1, e] == 1 if s < S - 1 else False
                states[s][e] = "exit" if not next_present else "stay"
    return states
