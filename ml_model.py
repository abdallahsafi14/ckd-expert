"""
╔══════════════════════════════════════════════════════════════════╗
║  Machine Learning Component — CKD Expert System                  ║
║  Model: Decision Tree Classifier                                 ║
║                                                                  ║
║  Role in Hybrid System:                                         ║
║  · Trained on synthetic CKD data (KDIGO 2024 guidelines)        ║
║  · Outputs: predicted stage + confidence per class              ║
║  · Combined with Fuzzy Engine for final hybrid diagnosis        ║
║                                                                  ║
║  Knowledge Acquisition Method:                                  ║
║  · Data generated from KDIGO medical thresholds                 ║
║  · Decision Tree extracts rules automatically from data         ║
║  · Replaces manual rule writing for initial classification      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATA GENERATION
# Based on KDIGO 2024 clinical thresholds
# ═══════════════════════════════════════════════════════════════════

def generate_ckd_dataset(n_samples=2000, random_state=42):
    """
    Generate synthetic CKD dataset based on KDIGO 2024 guidelines.
    Each sample: [gfr, creatinine, protein, bp, sugar, bun] → stage (0-5)

    Medical basis:
    - Stage 0 (Healthy):  GFR≥90, Cr<1.2, protein<30, BUN<20
    - Stage 1 (G1):       GFR≥90, protein 30-300 or Cr slightly elevated
    - Stage 2 (G2):       GFR 60-89 + biomarker abnormalities
    - Stage 3 (G3):       GFR 30-59 (any biomarker pattern)
    - Stage 4 (G4):       GFR 15-29 OR Cr>4.5 OR BUN>80
    - Stage 5 (G5):       GFR<15 OR Cr>8 OR BUN>115
    """
    rng = np.random.RandomState(random_state)

    X, y = [], []

    samples_per_stage = n_samples // 6

    # ── Stage 0: Healthy ──────────────────────────────────────────
    for _ in range(samples_per_stage):
        gfr  = rng.uniform(90, 120)
        cr   = rng.uniform(0.5, 1.1)
        pr   = rng.uniform(0, 28)
        bp   = rng.uniform(90, 122)
        su   = rng.uniform(65, 100)
        bun  = rng.uniform(7, 20)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(0)

    # ── Stage 1: G1 (GFR normal + kidney damage markers) ─────────
    for _ in range(samples_per_stage):
        gfr  = rng.uniform(90, 120)
        cr   = rng.uniform(1.0, 1.5)
        pr   = rng.uniform(30, 300)
        bp   = rng.uniform(118, 135)
        su   = rng.uniform(90, 130)
        bun  = rng.uniform(15, 30)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(1)

    # ── Stage 2: G2 (GFR 60-89) ──────────────────────────────────
    for _ in range(samples_per_stage):
        gfr  = rng.uniform(60, 89)
        cr   = rng.uniform(1.2, 2.0)
        pr   = rng.uniform(50, 500)
        bp   = rng.uniform(125, 148)
        su   = rng.uniform(100, 160)
        bun  = rng.uniform(20, 45)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(2)

    # ── Stage 3: G3 (GFR 30-59) ──────────────────────────────────
    for _ in range(samples_per_stage):
        gfr  = rng.uniform(30, 59)
        cr   = rng.uniform(1.8, 3.5)
        pr   = rng.uniform(200, 1500)
        bp   = rng.uniform(138, 162)
        su   = rng.uniform(130, 220)
        bun  = rng.uniform(40, 80)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(3)

    # ── Stage 4: G4 (GFR 15-29 OR critical biomarkers) ───────────
    for _ in range(samples_per_stage):
        variant = rng.randint(0, 3)
        if variant == 0:   # GFR-based
            gfr = rng.uniform(15, 29)
            cr  = rng.uniform(3.0, 6.0)
            pr  = rng.uniform(500, 2500)
            bp  = rng.uniform(152, 178)
            su  = rng.uniform(170, 270)
            bun = rng.uniform(70, 115)
        elif variant == 1:  # Cr-based override
            gfr = rng.uniform(70, 110)
            cr  = rng.uniform(5.0, 8.0)
            pr  = rng.uniform(300, 2000)
            bp  = rng.uniform(140, 175)
            su  = rng.uniform(120, 220)
            bun = rng.uniform(60, 110)
        else:               # BUN-based
            gfr = rng.uniform(60, 105)
            cr  = rng.uniform(2.5, 5.0)
            pr  = rng.uniform(1000, 4000)
            bp  = rng.uniform(148, 180)
            su  = rng.uniform(140, 260)
            bun = rng.uniform(80, 120)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(4)

    # ── Stage 5: G5 / ESRD ───────────────────────────────────────
    for _ in range(samples_per_stage):
        variant = rng.randint(0, 3)
        if variant == 0:   # GFR failure
            gfr = rng.uniform(0, 14)
            cr  = rng.uniform(6.0, 12.0)
            pr  = rng.uniform(1500, 5000)
            bp  = rng.uniform(170, 200)
            su  = rng.uniform(190, 380)
            bun = rng.uniform(100, 200)
        elif variant == 1:  # Cr critical override
            gfr = rng.uniform(60, 115)
            cr  = rng.uniform(9.0, 15.0)
            pr  = rng.uniform(500, 5000)
            bp  = rng.uniform(150, 200)
            su  = rng.uniform(130, 380)
            bun = rng.uniform(80, 200)
        else:               # Multi-critical
            gfr = rng.uniform(5, 50)
            cr  = rng.uniform(7.0, 15.0)
            pr  = rng.uniform(2000, 5000)
            bp  = rng.uniform(165, 200)
            su  = rng.uniform(200, 400)
            bun = rng.uniform(115, 200)
        X.append([gfr, cr, pr, bp, su, bun])
        y.append(5)

    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

# Feature names for interpretability
FEATURE_NAMES = ['GFR', 'Creatinine', 'Protein', 'BloodPressure', 'BloodSugar', 'BUN']
CLASS_NAMES   = ['Healthy', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5']

def train_decision_tree(random_state=42):
    """
    Train Decision Tree on KDIGO-based synthetic CKD data.
    Returns trained model + evaluation metrics.
    """
    X, y = generate_ckd_dataset(n_samples=3000, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Decision Tree — depth 8 balances accuracy and interpretability
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # handles any class imbalance
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    return model, acc, report, X_test, y_test


# ── Train at import time (fast: ~0.1 seconds) ─────────────────────
_MODEL, _ACCURACY, _REPORT, _X_TEST, _Y_TEST = train_decision_tree()


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════════

def ml_predict(gfr, creatinine, protein, bp, sugar, bun):
    """
    Run Decision Tree inference on a single patient.

    Returns:
        dict with:
          - predicted_stage   : int 0-5
          - confidence        : float 0-1 (max class probability)
          - class_probs       : dict {stage_name: probability}
          - feature_importance: dict {feature: importance_score}
    """
    X = np.array([[gfr, creatinine, protein, bp, sugar, bun]])

    predicted_stage = int(_MODEL.predict(X)[0])
    probs           = _MODEL.predict_proba(X)[0]

    class_probs = {
        CLASS_NAMES[i]: round(float(probs[i]), 3)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_stage":    predicted_stage,
        "confidence":         round(float(probs[predicted_stage]), 3),
        "class_probs":        class_probs,
        "feature_importance": {
            FEATURE_NAMES[i]: round(float(_MODEL.feature_importances_[i]), 3)
            for i in range(len(FEATURE_NAMES))
        },
        "model_accuracy": round(_ACCURACY, 3),
    }


def get_model_info():
    """Return model metadata for API and reporting."""
    return {
        "type":           "Decision Tree Classifier",
        "max_depth":      _MODEL.max_depth,
        "n_features":     len(FEATURE_NAMES),
        "n_classes":      len(CLASS_NAMES),
        "feature_names":  FEATURE_NAMES,
        "class_names":    CLASS_NAMES,
        "train_accuracy": round(_ACCURACY, 3),
        "n_leaves":       _MODEL.get_n_leaves(),
        "training_data":  "Synthetic — KDIGO 2024 guidelines",
    }


def get_tree_rules(max_depth=3):
    """Export top-level decision tree rules as readable text."""
    return export_text(
        _MODEL,
        feature_names=FEATURE_NAMES,
        max_depth=max_depth,
        show_weights=True,
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — SELF TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔" + "═"*62 + "╗")
    print("║  Decision Tree — CKD ML Model Test                          ║")
    print("╠" + "═"*62 + "╣")

    info = get_model_info()
    print(f"║  Model type  : {info['type']:46}║")
    print(f"║  Max depth   : {str(info['max_depth']):46}║")
    print(f"║  Leaves      : {str(info['n_leaves']):46}║")
    print(f"║  Accuracy    : {str(info['train_accuracy']):46}║")
    print("╠" + "═"*62 + "╣")

    cases = [
        ("Healthy",     105, 0.9,  12,   112,  85,  12,  0),
        ("Stage 1",     95,  1.1,  90,   120,  95,  15,  1),
        ("Stage 2",     72,  1.5,  180,  130, 120,  22,  2),
        ("Stage 3",     48,  2.2,  350,  145, 160,  50,  3),
        ("Stage 4",     22,  4.0, 1000,  158, 195,  85,  4),
        ("Stage 5",     8,   9.0, 3000,  182, 240, 145,  5),
        ("GFR ok+حرج", 97,  1.5, 3550,  163,  60, 169,  5),
    ]

    print(f"║  {'Case':20} {'Pred':7} {'Exp':7} {'Conf':8} {'✓':4}   ║")
    print("╠" + "═"*62 + "╣")

    passed = 0
    for name, gfr, cr, pr, bp, su, bu, exp in cases:
        r  = ml_predict(gfr, cr, pr, bp, su, bu)
        ok = r['predicted_stage'] == exp
        if ok: passed += 1
        sym = "✅" if ok else "❌"
        print(f"║  {name:20} Stage {r['predicted_stage']}  Stage {exp}  "
              f"{r['confidence']:.2f}     {sym}   ║")

    print("╠" + "═"*62 + "╣")
    print(f"║  Result: {passed}/{len(cases)} correct"
          + " " * (52 - len(f"  Result: {passed}/{len(cases)} correct")) + "║")
    print("╚" + "═"*62 + "╝")

    print()
    print("── Top Feature Importances ──")
    r0 = ml_predict(48, 2.2, 350, 145, 160, 50)
    for feat, imp in sorted(
        r0['feature_importance'].items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 30)
        print(f"  {feat:15} {bar:30} {imp:.3f}")

    print()
    print("── Sample Tree Rules (depth ≤ 3) ──")
    print(get_tree_rules(max_depth=3))