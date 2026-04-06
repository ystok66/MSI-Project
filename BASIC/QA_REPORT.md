================================================================================
RSA QA Test Suite
================================================================================

Running Test: Scalar Implicature
Description: RSA should prefer the object that doesn't match a stronger alternative.
----------------------------------------
Query: ['blue']
Object               | L0 Prob    | RSA Prob   | Diff (RSA-L0)
------------------------------------------------------------
blue_box             | 0.6140     | 0.5595     | -0.0545
blue_solid           | 0.3311     | 0.4405     | +0.1094
red_box              | 0.0549     | 0.0000     | -0.0549
----------------------------------------
Result: PASS: RSA slightly preferred blue_box.

================================================================================

Running Test: Mutual Exclusivity
Description: RSA should map novel word to novel object.
----------------------------------------
Query: ['dax']
Object               | L0 Prob    | RSA Prob   | Diff (RSA-L0)
------------------------------------------------------------
blue_box             | 0.4974     | 0.0000     | -0.4974
red_box              | 0.5026     | 1.0000     | +0.4974
----------------------------------------
Result: PASS: RSA mapped 'dax' to red_box (ME observed).

================================================================================

Running Test: Multi-Intent Overlap
Description: RSA should penalize using one object for multiple intents.
----------------------------------------
Query: 1 blue, 1 solid
Assignment                     | L0 Prob    | RSA Prob
------------------------------------------------------------
BlueBox, RedSolid              | 0.9999     | 1.0000
BlueSolid, RedSolid            | 0.0000     | 0.0000
----------------------------------------
Result: PASS: RSA boosted disjoint assignment (Ratio L0=589177.36, RSA=1000000000.00)

================================================================================

Running Test: Volume Penalty (Specificity)
Description: RSA/L0 should prefer specific terms for specific objects.
----------------------------------------
Query: 'blue' (vs 'navy')
Object               | L0 Prob    | RSA Prob   | Diff (RSA-L0)
------------------------------------------------------------
Center (Navy-ish)    | 0.9241     | 0.0000     | -0.9241
Offset (Blue-ish)    | 0.0759     | 1.0000     | +0.9241
----------------------------------------
Result: PASS: RSA boosted 'Offset' object (SI/Hyponymy observed).

================================================================================

Running Test: Ambiguity Resolution
Description: RSA should be sharper than L0.
----------------------------------------
Object               | L0 Prob    | RSA Prob   | Diff (RSA-L0)
------------------------------------------------------------
BlueBox              | 0.8366     | 0.9997     | +0.1631
RedBox               | 0.1634     | 0.0003     | -0.1631
----------------------------------------
Result: PASS: RSA is sharper/more confident.

================================================================================

SUMMARY
----------------------------------------
Scalar Implicature        : PASS
Mutual Exclusivity        : PASS
Multi-Intent Overlap      : PASS
Volume Penalty (Specificity) : PASS
Ambiguity Resolution      : PASS
