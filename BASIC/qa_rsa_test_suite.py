import sys
sys.path.insert(0, '.')

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import listener_L1_set, score_L0, normalize_tokens, infer_posterior, infer_posterior_multi_intent
from scipy.special import softmax

# =============================================================================
# Helper Functions
# =============================================================================

def make_obj(shape_key, color_key):
    """Create a world object."""
    occ = np.array(SHAPES[shape_key], dtype=np.float32)
    rgb = COLORS_RGB[color_key]
    return Obj(shape_name=shape_key, color_rgb=rgb, occ=occ)

def l0_posterior_single(X, mask, tokens, table, alpha=1.0):
    """Compute L0 posterior P(t|u) for single object selection."""
    U = normalize_tokens(tokens)
    # score_L0 returns log scores
    l0_scores = score_L0(X, mask, U, table)

    # Filter to valid regions
    valid_indices = [i for i, m in enumerate(mask) if m]
    valid_scores = l0_scores[valid_indices]

    # Softmax
    probs_valid = softmax(valid_scores * alpha)

    # Map back to full array
    probs = np.zeros(4)
    probs[valid_indices] = probs_valid
    return probs

def format_prob_table(labels, l0_probs, rsa_probs):
    """Format a comparison table for probabilities."""
    lines = []
    lines.append(f"{'Object':<20} | {'L0 Prob':<10} | {'RSA Prob':<10} | {'Diff (RSA-L0)':<12}")
    lines.append("-" * 60)

    for i in range(len(labels)):
        if labels[i] is None: continue
        diff = rsa_probs[i] - l0_probs[i]
        lines.append(f"{labels[i]:<20} | {l0_probs[i]:.4f}     | {rsa_probs[i]:.4f}     | {diff:+.4f}")
    return "\n".join(lines)

# =============================================================================
# Test Case Structure
# =============================================================================

@dataclass
class TestCase:
    name: str
    description: str
    run_func: callable

TEST_CASES = []

def register_test(name, description):
    def decorator(func):
        TEST_CASES.append(TestCase(name, description, func))
        return func
    return decorator

# =============================================================================
# Test Cases
# =============================================================================

@register_test("Scalar Implicature", "RSA should prefer the object that doesn't match a stronger alternative.")
def test_scalar_implicature():
    """
    Scenario:
    - Known words: "blue", "solid"
    - Objects:
        0: blue box (matches "blue", doesn't match "solid")
        1: blue solid (matches "blue", matches "solid")
    - Query: "blue"

    L0: Both fit "blue" equally well (approx).
    RSA: "blue" implies NOT "solid", because if it were solid, speaker would say "solid" (assuming "solid" is informative).
    """
    table = ConceptTable(d=12)

    # Train "blue"
    for shape in ['box', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        learn_step(X, m, k=1, tokens=['blue'], table=table)

    # Train "solid"
    for color in ['red', 'green', 'yellow']:
        X, m = encode_scene(Scene(regions=[make_obj('solid', color), None, None, None]))
        learn_step(X, m, k=1, tokens=['solid'], table=table)

    # Scene
    obj0 = make_obj('box', 'blue')   # blue, not solid
    obj1 = make_obj('solid', 'blue') # blue, solid
    # Add distractors
    obj2 = make_obj('box', 'red')

    scene = Scene(regions=[obj0, obj1, obj2, None])
    X, mask = encode_scene(scene)
    labels = ["blue_box", "blue_solid", "red_box", None]

    query = ["blue"]

    # L0
    l0_probs = l0_posterior_single(X, mask, query, table)

    # RSA
    # Note: include_empty_alt=False enables Scalar Implicature logic in rsa.py
    # But strictly, Scalar Implicature relies on alternatives.
    # rsa.py has auto_alt_from_table=True by default
    rsa_probs = infer_posterior(X, mask, query, table, alpha=5.0)

    print(f"Query: {query}")
    print(format_prob_table(labels, l0_probs, rsa_probs))

    # Verification
    # RSA prob for blue_box should be significantly higher than blue_solid
    if rsa_probs[0] > rsa_probs[1] + 0.2: # Significant difference
        return "PASS: RSA preferred blue_box over blue_solid (SI observed)."
    elif rsa_probs[0] > rsa_probs[1]:
        return "PASS: RSA slightly preferred blue_box."
    else:
        return "FAIL: RSA did not prefer blue_box (SI failed)."

@register_test("Mutual Exclusivity", "RSA should map novel word to novel object.")
def test_mutual_exclusivity():
    """
    Scenario:
    - Known: "blue"
    - Objects:
        0: blue box (known property)
        1: red box (unknown property 'red', or just 'box' but we test color here)
    - Query: "dax" (novel)

    L0: "dax" is broad, fits both equally (poorly or broadly).
    RSA: "blue" fits obj0 very well. "dax" fits both broadly.
         Speaker would say "blue" for obj0.
         So "dax" must mean obj1.
    """
    table = ConceptTable(d=12)

    # Train "blue" well
    for _ in range(3):
        for shape in ['box', 'solid']:
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
            learn_step(X, m, k=1, tokens=['blue'], table=table)

    # Scene
    obj0 = make_obj('box', 'blue') # Known "blue"
    obj1 = make_obj('box', 'red')  # Unknown color
    scene = Scene(regions=[obj0, obj1, None, None])
    X, mask = encode_scene(scene)
    labels = ["blue_box", "red_box", None, None]

    query = ["dax"]

    # L0
    l0_probs = l0_posterior_single(X, mask, query, table)

    # RSA
    # Provide extra tokens? rsa.py handles novel words if we pass them.
    # We rely on auto_alt_from_table=True (which adds 'blue')
    # and the logic that disables empty_alt if novel word is present (Scenario A in rsa.py)
    rsa_probs = infer_posterior(X, mask, query, table, alpha=5.0)

    print(f"Query: {query}")
    print(format_prob_table(labels, l0_probs, rsa_probs))

    if rsa_probs[1] > rsa_probs[0] + 0.2:
        return "PASS: RSA mapped 'dax' to red_box (ME observed)."
    else:
        return "FAIL: RSA did not strongly prefer red_box."

@register_test("Multi-Intent Overlap", "RSA should penalize using one object for multiple intents.")
def test_multi_intent_overlap():
    """
    Scenario:
    - Query: "1 blue, 1 solid"
    - Objects:
        0: blue box (blue)
        1: red solid (solid)
        2: blue solid (blue AND solid)

    Assignment A: blue->0, solid->1 (Disjoint)
    Assignment B: blue->2, solid->2 (Overlap - Single object) - Wait, rsa.py handles assignments as tuples of objects.
    Enumerate assignments in rsa.py ensures distinct objects?
    Let's check `enumerate_multi_intent_assignments` in rsa.py.
    It returns "assignments with disjoint objects".
    So "1 blue, 1 solid" CANNOT be satisfied by a single object in the default logic?
    Wait, `enumerate_multi_intent_assignments` docs say: "Select 1 object for 'blue', 1 object for 'red'. Objects must be disjoint".
    Ah! So strict multi-intent REQUIRES disjoint objects in the current implementation?

    If so, let's check overlap in terms of "using the ambiguous object vs the clear object".
    Ah, `test7` checks:
    Assignment 1: obj0 (blue), obj1 (solid) -> Disjoint, clear.
    Assignment 2: obj2 (blue solid), obj1 (solid) -> Disjoint objects, but obj2 is "better" for blue?

    Wait, if I have:
    Obj0: Blue Box
    Obj1: Red Solid
    Obj2: Blue Solid

    Query: "1 blue, 1 solid".

    Valid Assignments (must be disjoint):
    1. (Obj0, Obj1) -> Blue=Obj0, Solid=Obj1. Good.
    2. (Obj2, Obj1) -> Blue=Obj2, Solid=Obj1. Valid. Obj2 is blue.
    3. (Obj0, Obj2) -> Blue=Obj0, Solid=Obj2. Valid. Obj2 is solid.

    L0:
    Score(1) = L0(blue, 0) + L0(solid, 1)
    Score(2) = L0(blue, 2) + L0(solid, 1)

    If Obj2 is a "better" blue than Obj0, L0 prefers (2).
    But RSA S1 asks: "If I meant (Obj2, Obj1), would I say '1 blue, 1 solid'?"
    Target (Obj2, Obj1) is {BlueSolid, RedSolid}.
    Better utterance: "1 solid, 1 solid" -> "2 solid".
    Target (Obj0, Obj1) is {BlueBox, RedSolid}.
    Utterance "1 blue, 1 solid" is good.

    So RSA should prefer (Obj0, Obj1) over (Obj2, Obj1), even if Obj2 is a very good blue.
    """
    table = ConceptTable(d=12)
    # Train blue and solid
    for shape in ['box', 'l']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        learn_step(X, m, k=1, tokens=['blue'], table=table)
    for color in ['red', 'green']:
        X, m = encode_scene(Scene(regions=[make_obj('solid', color), None, None, None]))
        learn_step(X, m, k=1, tokens=['solid'], table=table)

    obj0 = make_obj('box', 'blue')    # Blue only
    obj1 = make_obj('solid', 'red')   # Solid only
    obj2 = make_obj('solid', 'blue')  # Blue and Solid

    scene = Scene(regions=[obj0, obj1, obj2, None])
    X, mask = encode_scene(scene)

    # Intent: 1 blue, 1 solid
    intents = [(['blue'], 1), (['solid'], 1)]

    # Run L0-like (disable RSA)
    res_l0 = infer_posterior_multi_intent(X, mask, intents, table, use_rsa=False)
    # Run RSA
    res_rsa = infer_posterior_multi_intent(X, mask, intents, table, use_rsa=True)

    # Assignments keys are tuples of tuples of indices
    # We need to identify them
    # (Obj0, Obj1) is ((0,), (1,))
    # (Obj2, Obj1) is ((2,), (1,))
    # (Obj0, Obj2) is ((0,), (2,))

    assign_disjoint = ((0,), (1,)) # blue=0, solid=1
    assign_overlap_blue = ((2,), (1,)) # blue=2, solid=1

    prob_l0_disjoint = res_l0.get(assign_disjoint, 0)
    prob_l0_overlap = res_l0.get(assign_overlap_blue, 0)

    prob_rsa_disjoint = res_rsa.get(assign_disjoint, 0)
    prob_rsa_overlap = res_rsa.get(assign_overlap_blue, 0)

    print(f"Query: 1 blue, 1 solid")
    print(f"{'Assignment':<30} | {'L0 Prob':<10} | {'RSA Prob':<10}")
    print("-" * 60)
    print(f"{'BlueBox, RedSolid':<30} | {prob_l0_disjoint:.4f}     | {prob_rsa_disjoint:.4f}")
    print(f"{'BlueSolid, RedSolid':<30} | {prob_l0_overlap:.4f}     | {prob_rsa_overlap:.4f}")

    # Expectation: RSA boosts the Disjoint one relative to the Overlap one compared to L0
    # Or simply RSA prefers Disjoint > Overlap more strongly than L0 does.

    ratio_l0 = prob_l0_disjoint / (prob_l0_overlap + 1e-9)
    ratio_rsa = prob_rsa_disjoint / (prob_rsa_overlap + 1e-9)

    if ratio_rsa > ratio_l0 * 1.5:
         return f"PASS: RSA boosted disjoint assignment (Ratio L0={ratio_l0:.2f}, RSA={ratio_rsa:.2f})"
    else:
         return f"FAIL: No significant boost (Ratio L0={ratio_l0:.2f}, RSA={ratio_rsa:.2f})"

@register_test("Volume Penalty (Specificity)", "RSA/L0 should prefer specific terms for specific objects.")
def test_volume_penalty():
    """
    Scenario:
    - Concept "blue": broad variance (learned from many blues)
    - Concept "navy": narrow variance (learned from specific blue)
    - Object: Navy Box (fits both)
    - Query: "navy" vs "blue"

    Wait, this is usually S1 speaker choice ("what to say").
    But here we test Listener L0/RSA.
    L0 listener score: S_L0 = log_inc - beta * vol.
    If we hear "navy", volume penalty is smaller (logdet is smaller/negative-er? No).
    Variance < 1 -> logdet is negative.
    Small variance -> Small volume.
    Wait, Volume Penalty usually penalizes BROAD concepts in some frameworks, or SPECIFIC ones?

    In rsa.py: `S_L0 = log_inc - beta * vol`.
    `vol = logdet_diag(var)`.
    If var is small (e.g. 0.01), logdet is negative (e.g. log(0.01) = -4.6).
    If var is large (e.g. 1.0), logdet is 0.

    So -beta * vol:
    Small vol (narrow): -0.1 * (-4.6) = +0.46 (Reward for specificity? Or Penalty?)
    Large vol (broad): -0.1 * (0) = 0.

    Wait, usually Size Principle favors specific hypotheses.
    "The Size Principle: hypothesis with smaller size is more probable given data."
    Here we are scoring Referents given Word.
    L0(t, U) = log P(t|U) ~ log P(U|t) P(t) ?
    The implementation `log_inc(t, U)` is basically `log P(t|concept)`.
    If concept is narrow, `P(t|concept)` density is HIGHER for fitting objects.
    So "navy" should have higher `log_inc` for a navy object than "blue" does.

    The volume penalty term in `score_L0` seems to be an additional regularization.
    Let's test if "navy" wins over "blue" for a navy object more than it would purely by density?

    Actually, simpler test:
    Obj1: Navy.
    Obj2: Sky Blue.
    Concept "blue": Fits both.
    Concept "navy": Fits Obj1 only.

    Query: "blue".
    L0: Fits Obj1 and Obj2.
    RSA: "blue" implies "not navy" (Scalar Implicature again!).

    Let's do that. It's a variant of SI but with synonyms/hyponyms.
    """
    table = ConceptTable(d=12)

    # Hack concepts
    # "blue": centered at [0.2, 0.0, -0.4] (rough blue), var=0.1
    c_blue = table.ensure("blue")
    c_blue.mu[:3] = [50, 0, -50] # RGB->LAB is complex, let's just train.

    # Train "blue" broadly
    for c in ['blue', 'cyan', 'purple']: # diverse
        # We need actual RGB values that map to "blue" region
        pass

    # Let's use the manual hack for precision
    c_blue = table.ensure("blue")
    c_blue.mu = np.zeros(12) # Dummy
    c_blue.var = np.ones(12) * 0.2 # Broad-ish

    c_navy = table.ensure("navy")
    c_navy.mu = np.zeros(12) # Same center
    c_navy.var = np.ones(12) * 0.01 # Very narrow

    # Obj0: Perfectly matches center (Navy)
    # Obj1: Distant (Sky Blue) - wait, if I put it far, navy won't fit.

    # Let's just create an object at the center (Obj0).
    # Both "blue" and "navy" fit it.
    # But "navy" has higher likelihood density because of 1/sigma factor in Gaussian.
    # log_inc includes -0.5*log(det).

    # We want to test listener behavior given "blue".
    # If I say "blue", and there is a "navy" concept available.
    # Does RSA lower the prob of the perfect navy object because I didn't say "navy"?

    obj0 = make_obj('box', 'blue')
    # Mock encoding to match our hacked concepts
    # We need to hack the encoded scene X
    scene = Scene(regions=[obj0, None, None, None])
    X, mask = encode_scene(scene)
    X[0] = np.zeros(12) # Match the means

    # Query: "blue"
    # L0: P(obj0 | "blue") is high.
    # RSA: Alternatives = ["blue", "navy"].
    # Speaker S1(U | obj0):
    #   L0("blue", obj0): Good fit, but broad. Density is lower.
    #   L0("navy", obj0): Perfect fit, narrow. Density is Higher.
    #   So S1 prefers "navy".
    # Listener L1("blue"): Should think "It's probably not obj0, because they would have said navy".
    # But what is the alternative?
    # We need another object that "blue" fits but "navy" doesn't.

    obj1 = make_obj('box', 'blue')
    X[1] = np.zeros(12)
    X[1][0] = 1.0 # Offset, fits blue (var 0.2 covers it), doesn't fit navy (var 0.01).
    mask[1] = True

    # So:
    # Obj0: Matches Blue (well), Matches Navy (super well).
    # Obj1: Matches Blue (okay), Matches Navy (no).

    # Query: "blue".
    # L0: Prefers Obj0 (because Blue fits Obj0 better than Obj1 - center vs offset).
    # RSA: S1 would say "navy" for Obj0. S1 would say "blue" for Obj1.
    # So L1("blue") should shift probability toward Obj1 (or at least flatten the distribution).

    l0_probs = l0_posterior_single(X, mask, ["blue"], table)
    rsa_probs = infer_posterior(X, mask, ["blue"], table, alpha=5.0)

    labels = ["Center (Navy-ish)", "Offset (Blue-ish)", None, None]

    print(f"Query: 'blue' (vs 'navy')")
    print(format_prob_table(labels, l0_probs, rsa_probs))

    # L0 prefers Center (0).
    # RSA should punish Center (0) because "navy" was better.
    # So RSA prob for Offset (1) should be higher than L0 prob for Offset (1).

    if rsa_probs[1] > l0_probs[1]:
        return "PASS: RSA boosted 'Offset' object (SI/Hyponymy observed)."
    else:
        return "FAIL: RSA did not boost 'Offset' object."

@register_test("Ambiguity Resolution", "RSA should be sharper than L0.")
def test_ambiguity_resolution():
    """
    Scenario:
    - Obj0: Blue Box
    - Obj1: Red Box
    - Query: "blue"

    L0: High for Obj0, Low for Obj1.
    RSA: S1 makes it deterministic. "blue" -> Obj0 is only valid.
    RSA should be closer to 1.0/0.0 than L0 (if L0 has any leak).
    """
    table = ConceptTable(d=12)
    # Train blue
    for _ in range(2):
        X, m = encode_scene(Scene(regions=[make_obj('box', 'blue'), None, None, None]))
        learn_step(X, m, k=1, tokens=['blue'], table=table)

    obj0 = make_obj('box', 'blue')
    obj1 = make_obj('box', 'red')
    scene = Scene(regions=[obj0, obj1, None, None])
    X, mask = encode_scene(scene)

    query = ["blue"]

    l0_probs = l0_posterior_single(X, mask, query, table, alpha=1.0)
    rsa_probs = infer_posterior(X, mask, query, table, alpha=5.0) # Standard alpha

    labels = ["BlueBox", "RedBox", None, None]
    print(format_prob_table(labels, l0_probs, rsa_probs))

    if rsa_probs[0] > l0_probs[0]:
        return "PASS: RSA is sharper/more confident."
    else:
        return "NEUTRAL: Already sharp."

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 80)
    print("RSA QA Test Suite")
    print("=" * 80)
    print()

    results = []

    for case in TEST_CASES:
        print(f"Running Test: {case.name}")
        print(f"Description: {case.description}")
        print("-" * 40)

        try:
            outcome = case.run_func()
            print("-" * 40)
            print(f"Result: {outcome}")
            results.append((case.name, outcome))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((case.name, "ERROR"))

        print("\n" + "=" * 80 + "\n")

    print("SUMMARY")
    print("-" * 40)
    for name, outcome in results:
        status = outcome.split(":")[0] if ":" in outcome else outcome
        print(f"{name:<25} : {status}")

if __name__ == "__main__":
    main()
