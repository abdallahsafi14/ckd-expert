"""
╔══════════════════════════════════════════════════════════════════╗
║  Hybrid Diagnosis — ML + Fuzzy Expert System                    ║
║                                                                  ║
║  Architecture (as described by Dr. Reham Mansour):             ║
║  "محرك استدلال منطقي، ولكن بعض مدخلاته تأتي من نموذج تعلم آلي" ║
║                                                                  ║
║  Flow:                                                           ║
║  Input → [Decision Tree ML] ─────────────────┐                 ║
║       → [Fuzzy Engine Rules] ────────────────┤                 ║
║                                    ↓          │                 ║
║                           [Hybrid Combiner]   │                 ║
║                                    ↓                            ║
║                            Final Stage + Confidence             ║
║                                                                  ║
║  Combination Strategy:                                          ║
║  · If ML and Fuzzy AGREE  → use shared result (high confidence) ║
║  · If ML and Fuzzy DIFFER → weighted blend (fuzzy-heavy)        ║
║  · Critical override: if either detects stage 4/5, use max      ║
╚══════════════════════════════════════════════════════════════════╝
"""

from fuzzy_engine import diagnose as fuzzy_diagnose
from ml_model    import ml_predict


# ── Hybrid weights ────────────────────────────────────────────────
# Fuzzy engine carries more weight because it encodes expert rules
# ML carries weight as a second opinion / knowledge acquisition source
FUZZY_WEIGHT = 0.65
ML_WEIGHT    = 0.35


def hybrid_diagnose(gfr_val, creatinine_val, protein_val,
                    bp_val, sugar_val, bun_val):
    """
    Hybrid CKD diagnosis combining Decision Tree ML + Fuzzy Expert System.

    Returns structured result with:
    - fuzzy_result   : raw fuzzy engine output
    - ml_result      : raw decision tree output
    - hybrid_stage   : final combined stage
    - hybrid_score   : combined confidence/score
    - agreement      : whether ML and Fuzzy agree
    - method         : which combination strategy was used
    - explanation    : human-readable explanation of the decision
    """

    # ── Step 1: Run both engines independently ────────────────────
    fuzzy = fuzzy_diagnose(
        gfr_val, creatinine_val, protein_val,
        bp_val, sugar_val, bun_val
    )
    ml = ml_predict(
        gfr_val, creatinine_val, protein_val,
        bp_val, sugar_val, bun_val
    )

    fuzzy_stage = fuzzy["stage"]
    ml_stage    = ml["predicted_stage"]
    ml_conf     = ml["confidence"]
    fuzzy_score = fuzzy["score"]

    # ── Step 2: Determine combination strategy ────────────────────

    # CRITICAL SAFETY RULE: if either engine detects stage 4 or 5,
    # never downgrade — patient safety takes priority
    critical_stage = max(fuzzy_stage, ml_stage)

    if fuzzy_stage >= 4 or ml_stage >= 4:
        # Critical override — take the maximum (worst case)
        final_stage = critical_stage
        method      = "critical_override"
        explanation = (
            f"تم تفعيل قاعدة الأمان: أحد النموذجين كشف مرحلة حرجة "
            f"(Fuzzy=مرحلة {fuzzy_stage}, ML=مرحلة {ml_stage}). "
            f"النظام يأخذ الأسوأ (مرحلة {final_stage}) لأسباب أمان طبية."
        )
        hybrid_score = max(fuzzy_score / 5.0, ml_conf)

    elif fuzzy_stage == ml_stage:
        # Full agreement — high confidence result
        final_stage  = fuzzy_stage
        method       = "full_agreement"
        explanation  = (
            f"المحرك الفازي والـ Decision Tree كلاهما اتفقا على مرحلة {final_stage}. "
            f"ثقة عالية في النتيجة."
        )
        hybrid_score = (fuzzy_score / 5.0) * FUZZY_WEIGHT + ml_conf * ML_WEIGHT

    elif abs(fuzzy_stage - ml_stage) == 1:
        # Minor disagreement (adjacent stages) — weighted blend
        blended = fuzzy_stage * FUZZY_WEIGHT + ml_stage * ML_WEIGHT
        final_stage = round(blended)
        method      = "weighted_blend"
        explanation = (
            f"خلاف طفيف بين النموذجين: Fuzzy=مرحلة {fuzzy_stage}, ML=مرحلة {ml_stage}. "
            f"تم الدمج الموزون (Fuzzy {int(FUZZY_WEIGHT*100)}% + ML {int(ML_WEIGHT*100)}%) "
            f"→ مرحلة {final_stage}."
        )
        hybrid_score = (fuzzy_score / 5.0) * FUZZY_WEIGHT + ml_conf * ML_WEIGHT

    else:
        # Major disagreement — defer to fuzzy (expert rules)
        final_stage  = fuzzy_stage
        method       = "fuzzy_priority"
        explanation  = (
            f"خلاف كبير: Fuzzy=مرحلة {fuzzy_stage}, ML=مرحلة {ml_stage}. "
            f"النظام يعتمد على المحرك الفازي (القواعد الطبية الخبيرة) في حالة الخلاف الكبير."
        )
        hybrid_score = fuzzy_score / 5.0

    agreement = (fuzzy_stage == ml_stage)

    # ── Step 3: Build final output ────────────────────────────────
    # Use fuzzy engine's metadata (labels, recommendations) for final stage
    stage_meta = _get_stage_metadata(final_stage)

    return {
        # ── Hybrid result ──
        "hybrid_stage":     final_stage,
        "hybrid_score":     round(hybrid_score, 3),
        "agreement":        agreement,
        "method":           method,
        "explanation":      explanation,
        "label":            stage_meta["label"],
        "label_en":         stage_meta["label_en"],
        "risk":             stage_meta["risk"],
        "color":            stage_meta["color"],
        "recommendations":  stage_meta["recommendations"],

        # ── Individual engine results ──
        "fuzzy": {
            "stage":       fuzzy_stage,
            "score":       fuzzy["score"],
            "activations": fuzzy["activations"],
        },
        "ml": {
            "stage":              ml_stage,
            "confidence":         ml_conf,
            "class_probs":        ml["class_probs"],
            "feature_importance": ml["feature_importance"],
            "model_accuracy":     ml["model_accuracy"],
        },

        # ── Input echo ──
        "inputs": fuzzy["inputs"],
    }


def _get_stage_metadata(stage):
    """Return labels and recommendations for a given stage."""
    meta = {
        0: {
            "label":    "طبيعي — لا يوجد مرض كلوي",
            "label_en": "Healthy — No CKD",
            "risk":     "منخفض",
            "color":    "green",
            "recommendations": [
                "استمر في نمط الحياة الصحي",
                "أجرِ فحصاً دورياً كل سنة",
                "اشرب 6-8 أكواب ماء يومياً",
                "تجنب NSAIDs",
            ],
        },
        1: {
            "label":    "المرحلة الأولى — تلف كلوي مع وظيفة طبيعية",
            "label_en": "Stage G1 — Kidney Damage, Normal GFR (≥90)",
            "risk":     "منخفض إلى متوسط",
            "color":    "yellow",
            "recommendations": [
                "مراجعة طبيب كل 3 أشهر",
                "تحكم في ضغط الدم (< 130/80)",
                "قلل الصوديوم والبروتين الزائد",
                "أوقف التدخين",
            ],
        },
        2: {
            "label":    "المرحلة الثانية — انخفاض خفيف في وظيفة الكلى",
            "label_en": "Stage G2 — Mildly Decreased GFR (60-89)",
            "risk":     "متوسط",
            "color":    "orange",
            "recommendations": [
                "متابعة مع أخصائي كلى كل 3 أشهر",
                "نظام غذائي منخفض البوتاسيوم والفسفور",
                "تحكم صارم في ضغط الدم والسكر",
                "تجنب NSAIDs",
            ],
        },
        3: {
            "label":    "المرحلة الثالثة — انخفاض معتدل في وظيفة الكلى",
            "label_en": "Stage G3 — Moderately Decreased GFR (30-59)",
            "risk":     "متوسط إلى عالٍ",
            "color":    "red-light",
            "recommendations": [
                "إحالة فورية لأخصائي أمراض الكلى",
                "نظام غذائي كلوي متخصص",
                "مراقبة البوتاسيوم والفوسفور والكالسيوم",
                "تقييم صحة القلب",
            ],
        },
        4: {
            "label":    "المرحلة الرابعة — انخفاض حاد في وظيفة الكلى",
            "label_en": "Stage G4 — Severely Decreased GFR (15-29)",
            "risk":     "عالٍ جداً",
            "color":    "red",
            "recommendations": [
                "إشراف طبي مكثف شهري",
                "التخطيط لغسيل الكلى أو زراعة كلية",
                "نظام غذائي صارم",
                "مراقبة ضغط الدم يومياً",
            ],
        },
        5: {
            "label":    "المرحلة الخامسة — فشل كلوي (ESRD)",
            "label_en": "Stage G5 — Kidney Failure / ESRD (GFR < 15)",
            "risk":     "حرج",
            "color":    "critical",
            "recommendations": [
                "⚠️ حالة طارئة — تدخل طبي فوري",
                "غسيل الكلى عاجل",
                "تقييم للزراعة الكلوية",
                "علاج مكثف لجميع المضاعفات",
            ],
        },
    }
    return meta.get(stage, meta[0])


# ── Self test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("طبيعي",          105, 0.9,  12,   112,  85,  12,  0),
        ("مرحلة 1",        95,  1.1,  90,   120,  95,  15,  1),
        ("مرحلة 2",        72,  1.5,  180,  130, 120,  22,  2),
        ("مرحلة 3",        48,  2.2,  350,  145, 160,  50,  3),
        ("مرحلة 4",        22,  4.0,  1000, 158, 195,  85,  4),
        ("فشل كلوي",       8,   9.0,  3000, 182, 240, 145,  5),
        ("GFR طبيعي+حرج",  97,  1.5,  3550, 163,  60, 169,  5),
    ]

    print()
    print("╔" + "═"*80 + "╗")
    print("║  Hybrid System Test — Fuzzy + Decision Tree                                  ║")
    print("╠" + "═"*80 + "╣")
    print(f"║  {'الحالة':20} {'Fuzzy':8} {'ML':8} {'Hybrid':8} {'Exp':6} {'Method':20} {'✓':3} ║")
    print("╠" + "═"*80 + "╣")

    passed = 0
    for name, *vals, exp in cases:
        r  = hybrid_diagnose(*vals)
        ok = r["hybrid_stage"] == exp
        if ok: passed += 1
        sym = "✅" if ok else "❌"
        print(f"║  {name:20} مرحلة {r['fuzzy']['stage']}   مرحلة {r['ml']['stage']}   "
              f"مرحلة {r['hybrid_stage']}   مرحلة {exp}   {r['method']:20} {sym}  ║")

    print("╠" + "═"*80 + "╣")
    status = "✅ جميع الحالات صحيحة!" if passed == len(cases) else f"❌ {passed}/{len(cases)}"
    print(f"║  النتيجة: {status:70}║")
    print("╚" + "═"*80 + "╝")