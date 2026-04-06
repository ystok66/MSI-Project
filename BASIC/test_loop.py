"""Test multi-loop training effect on L0 vs RSA."""
import sys; sys.path.insert(0,'.')
import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step, scale_by_cardinality, update_concept_soft_counts
from rsa import infer_posterior_multi_intent, score_L0_multi_intent, enumerate_multi_intent_assignments, normalize_tokens, logsumexp, score_L0

def make_obj(s,c): 
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], occ=np.array(SHAPES[s], dtype=np.float32))

def l0_post(X, m, tok, t):
    U = normalize_tokens(tok)
    L0 = score_L0(X, m, U, t)
    e = np.exp(L0 - np.max(L0[m]))
    e[~m] = 0
    return e / np.sum(e)

def learn_l0(X, m, k, tok, t):
    p = l0_post(X, m, tok, t)
    w = scale_by_cardinality(p, k, m)
    for token in tok:
        update_concept_soft_counts(t.ensure(token), X, m, w)

print("Loop | Method | blue_box | red_box | blue_solid | red_solid")
print("-" * 60)

for n_loop in [1, 3, 5, 10]:
    for method in ['L0', 'RSA']:
        t = ConceptTable(d=12)
        
        for _ in range(n_loop):
            X1, m1 = encode_scene(Scene(regions=[make_obj('box','blue'), None, None, None]))
            X2, m2 = encode_scene(Scene(regions=[make_obj('box','red'), None, None, None]))
            X3, m3 = encode_scene(Scene(regions=[make_obj('solid','blue'), None, None, None]))
            X4, m4 = encode_scene(Scene(regions=[make_obj('solid','red'), None, None, None]))
            
            if method == 'L0':
                learn_l0(X1, m1, 1, ['b','box'], t)
                learn_l0(X2, m2, 1, ['r','box'], t)
                learn_l0(X3, m3, 1, ['b','solid'], t)
                learn_l0(X4, m4, 1, ['r','solid'], t)
            else:
                learn_step(X1, m1, k=1, tokens=['b','box'], table=t)
                learn_step(X2, m2, k=1, tokens=['r','box'], table=t)
                learn_step(X3, m3, k=1, tokens=['b','solid'], table=t)
                learn_step(X4, m4, k=1, tokens=['r','solid'], table=t)
        
        # Test
        Xt, mt = encode_scene(Scene(regions=[
            make_obj('box','blue'), make_obj('box','red'), 
            make_obj('solid','blue'), make_obj('solid','red')
        ]))
        
        intents = [(['b'], 1), (['solid'], 1)]
        
        if method == 'L0':
            norm_intents = [(normalize_tokens(tok), cnt) for tok, cnt in intents]
            assignments = enumerate_multi_intent_assignments(mt, norm_intents)
            logits = np.array([score_L0_multi_intent(Xt, mt, norm_intents, a, t) for a in assignments])
            lse = logsumexp(logits)
            result = {a: np.exp(logits[i] - lse) for i, a in enumerate(assignments)}
        else:
            result = infer_posterior_multi_intent(Xt, mt, intents, t)
        
        # Marginal probabilities
        m = [0, 0, 0, 0]
        for a, p in result.items():
            for i in a[0]: m[i] += p
            for i in a[1]: m[i] += p
        
        print(f"{n_loop:4d} | {method:3s}  | {m[0]:.3f}    | {m[1]:.3f}   | {m[2]:.3f}      | {m[3]:.3f}")
