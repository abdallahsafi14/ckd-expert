"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  Fuzzy Expert System — Chronic Kidney Disease (CKD) Detection v3.0      ║
║                                                                           ║
║  Medical basis:                                                           ║
║  · KDIGO 2024 Clinical Practice Guideline (GFR G1-G5, Albumin A1-A3)   ║
║  · NIH-NCBI NBK305: BUN normal 7-20, Creatinine 0.6-1.2 (male)         ║
║  · SiPhox/AsianHeart: Cr Stage3 ≈ 1.6-3.0, Stage4 ≈ 3-8, Stage5 > 8   ║
║  · NKF-Florida: BUN uremia threshold > 80-100 mg/dL                     ║
║  · ADA: Fasting glucose: normal <100, prediabetes 100-125, DM ≥126      ║
║  · AHA/ACC 2017: Systolic BP: normal <120, crisis ≥180 mmHg             ║
║  · Proteinuria: micro 30-300, macro 300-3500, nephrotic >3500 mg/day    ║
║                                                                           ║
║  Architecture:                                                            ║
║  · 6 input variables × 4-5 fuzzy sets each                               ║
║  · 65 IF-THEN rules organized in 5 categories                            ║
║  · Weighted centroid + critical-stage blend defuzzification               ║
║  · Pure NumPy — no scikit-fuzzy — Python 3.14+ compatible                ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

INF = 1e9  # Open-ended upper bound for membership functions

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MEMBERSHIP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def trimf(x, a, b, c):
    """Triangular: rises a→b, falls b→c"""
    x = float(x)
    if x < a or x > c: return 0.0
    if x <= b: return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

def trapmf(x, a, b, c, d):
    """
    Trapezoidal: rises a→b, plateau b→c, falls c→d.
    Set d=INF for open-ended upper sets (e.g. 'critical' creatinine).
    Fixed boundary: x==d returns 0 only if d < INF.
    """
    x = float(x)
    if x < a: return 0.0
    if d >= INF:  # open-ended: anything >= c is fully in set
        if x >= INF: return 1.0
        if x <= b: return (x - a) / (b - a) if b != a else 1.0
        if x <= c: return 1.0
        return 1.0  # plateau extends to infinity
    if x > d: return 0.0
    if x <= b: return (x - a) / (b - a) if b != a else 1.0
    if x <= c: return 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INPUT VARIABLE FUZZY SETS
# ═══════════════════════════════════════════════════════════════════════════════

def gfr_sets(v):
    """
    Glomerular Filtration Rate (mL/min/1.73m²) — KDIGO 2024
    G1 Normal:    ≥ 90       G2 Mild:    60-89
    G3 Moderate:  30-59      G4 Severe:  15-29
    G5 Failure:   < 15
    """
    return {
        "normal":   trapmf(v,  90, 105, INF, INF),
        "mild":     trapmf(v,  55,  65,  85,  95),
        "moderate": trapmf(v,  25,  32,  52,  62),
        "severe":   trapmf(v,   8,  15,  24,  30),
        "failure":  trapmf(v,  -1,  -1,   9,  15),
    }

def creatinine_sets(v):
    """
    Serum Creatinine (mg/dL)
    Normal:    ≤ 1.2        Elevated:  1.2 - 2.5  (Stage 1-2 territory)
    High:      2.0 - 5.5   (Stage 3-4)  Very_high: 4.5 - 9.5  (Stage 4-5)
    Critical:  ≥ 8.0       (Stage 5 / uremia — definitive kidney failure)
    Sources: NIH-NCBI, SiPhox, AsianHeartInstitute, TabbakKidney
    Note: sets overlap intentionally for smooth fuzzy transitions
    """
    return {
        "normal":    trapmf(v, -1,  -1,  0.9,  1.2),
        "elevated":  trapmf(v,  1.0, 1.4, 2.0,  2.5),   # Stage 1-2
        "high":      trapmf(v,  1.8, 2.5, 4.5,  5.5),   # Stage 2-3-4
        "very_high": trapmf(v,  4.5, 6.0, 8.5, 10.0),   # Stage 4-5
        "critical":  trapmf(v,  8.0, 9.5, INF,  INF),   # Stage 5 / uremia
    }

def protein_sets(v):
    """
    24h Urine Protein (mg/day) — KDIGO 2024 Albuminuria Categories
    A1 Normal:    < 30        A2 Micro:  30 - 300
    A3 Macro:     300 - 3500  Nephrotic: > 3500  (massive kidney damage)
    """
    return {
        "normal":    trapmf(v,  -1,   -1,   20,   30),
        "micro":     trapmf(v,  25,   60,  280,  320),
        "macro":     trapmf(v, 280,  500, 3200, 3600),
        "nephrotic": trapmf(v, 3200, 3600, INF,  INF),
    }

def bp_sets(v):
    """
    Systolic Blood Pressure (mmHg) — AHA/ACC 2017 Guidelines
    Normal: < 120    Elevated: 120-129    Stage1 HTN: 130-139
    Stage2 HTN: 140-179    Crisis: ≥ 180
    """
    return {
        "normal":   trapmf(v,  -1,  -1,  112, 122),
        "elevated": trapmf(v, 118, 124,  128, 132),
        "stage1":   trapmf(v, 128, 133,  137, 142),
        "stage2":   trapmf(v, 138, 143,  172, 178),
        "crisis":   trapmf(v, 172, 182,  INF, INF),
    }

def sugar_sets(v):
    """
    Fasting Blood Sugar (mg/dL) — ADA Guidelines 2024
    Normal: 70-99    Prediabetes: 100-125    Diabetes: ≥ 126
    Uncontrolled: ≥ 200  (high cardiovascular + renal risk)
    """
    return {
        "normal":       trapmf(v,  60,  70,  95, 102),
        "prediabetes":  trapmf(v,  97, 107, 120, 128),
        "diabetes":     trapmf(v, 122, 130, 192, 202),
        "uncontrolled": trapmf(v, 195, 210, INF, INF),
    }

def bun_sets(v):
    """
    Blood Urea Nitrogen (mg/dL) — NIH-NCBI NBK305 / NKF
    Normal: 7-20    Mild: 21-40 (non-specific — may reflect dehydration/diet)
    Moderate: 40-80 (significant renal impairment likely)
    Severe: 80-120  (uremia — serious kidney disease)
    Uremic: > 115   (life-threatening — stage 5 territory)
    IMPORTANT: BUN alone is NON-SPECIFIC; interpret with creatinine ratio
    """
    return {
        "normal":   trapmf(v,  -1,  -1,  16,  22),
        "mild":     trapmf(v,  18,  25,  38,  45),
        "moderate": trapmf(v,  40,  52,  75,  85),
        "severe":   trapmf(v,  78,  95, 118, 128),
        "uremic":   trapmf(v, 115, 130, INF,  INF),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INFERENCE ENGINE
# 65 rules in 5 categories. Logic: AND = min, OR = max
# ═══════════════════════════════════════════════════════════════════════════════

def AND(*a): return min(a)
def OR(*a):  return max(a)

def evaluate_rules(g, cr, pr, bp, su, bu):
    act = {k: 0.0 for k in
           ["healthy","stage1","stage2","stage3","stage4","stage5"]}

    def fire(s, v): act[s] = max(act[s], v)

    # ─── CATEGORY A: HEALTHY CONFIRMATION ────────────────────────────────────
    # All markers normal → definitely healthy
    fire("healthy", AND(g["normal"],  cr["normal"], pr["normal"],
                        bp["normal"], su["normal"], bu["normal"]))
    fire("healthy", AND(g["normal"],  cr["normal"], pr["normal"], bu["normal"]))
    fire("healthy", AND(g["normal"],  cr["normal"], pr["normal"], bp["normal"]))
    fire("healthy", AND(g["mild"],    cr["normal"], pr["normal"], bu["normal"],
                        bp["normal"]))
    fire("healthy", AND(g["normal"],  cr["normal"], pr["normal"],
                        su["normal"]))

    # ─── CATEGORY B: GFR-DRIVEN STAGING (KDIGO Baseline) ─────────────────────
    # G2 (60-89) → Stage 1-2
    fire("stage1", AND(g["mild"],     cr["normal"], pr["micro"]))
    # removed: g_mild+cr_normal+bu_normal alone does not indicate stage1
    fire("stage2", AND(g["mild"],     cr["elevated"]))
    fire("stage2", AND(g["mild"],     pr["macro"]))
    fire("stage2", AND(g["mild"],     bu["moderate"]))
    fire("stage2", AND(g["mild"],     su["diabetes"], cr["elevated"]))
    fire("stage2", AND(g["mild"],     su["diabetes"], pr["micro"]))

    # G3 (30-59) → Stage 3
    fire("stage3", AND(g["moderate"], cr["normal"]))
    fire("stage3", AND(g["moderate"], cr["elevated"]))
    fire("stage3", AND(g["moderate"], pr["micro"]))
    fire("stage3", AND(g["moderate"], bu["mild"]))
    fire("stage3", AND(g["moderate"], cr["high"],     pr["macro"]))
    fire("stage3", AND(g["moderate"], bu["moderate"], cr["elevated"]))
    fire("stage3", AND(g["moderate"], bp["stage2"],   cr["elevated"]))
    fire("stage3", AND(g["moderate"], su["diabetes"], pr["macro"]))
    fire("stage3", AND(g["moderate"], bp["crisis"],   bu["mild"]))

    # G4 (15-29) → Stage 4
    fire("stage4", AND(g["severe"],   cr["elevated"]))
    fire("stage4", AND(g["severe"],   cr["high"]))
    fire("stage4", AND(g["severe"],   bu["moderate"]))
    fire("stage4", AND(g["severe"],   pr["macro"]))
    fire("stage4", AND(g["severe"],   cr["high"],     bu["moderate"]))
    fire("stage4", AND(g["severe"],   cr["very_high"],bu["moderate"]))
    fire("stage4", AND(g["severe"],   bp["crisis"],   cr["high"]))
    fire("stage4", AND(g["severe"],   su["diabetes"], pr["nephrotic"]))

    # G5 (< 15) → Stage 5
    fire("stage5", g["failure"])
    fire("stage5", AND(g["failure"],  cr["high"]))
    fire("stage5", AND(g["failure"],  cr["very_high"], bu["severe"]))
    fire("stage5", AND(g["failure"],  pr["nephrotic"], bu["severe"]))
    fire("stage5", AND(g["failure"],  bp["crisis"],    cr["very_high"]))

    # ─── CATEGORY C: NORMAL GFR + ABNORMAL BIOMARKERS ────────────────────────
    # Stage 1: kidney damage markers with preserved GFR (KDIGO definition)
    fire("stage1", AND(g["normal"],   pr["micro"]))
    fire("stage1", AND(g["normal"],   cr["elevated"], pr["normal"]))
    fire("stage1", AND(g["normal"],   bu["mild"],     cr["elevated"]))
    fire("stage1", AND(g["normal"],   bp["stage1"],   pr["micro"]))

    # Stage 2: multiple moderate abnormalities with normal/mild GFR
    fire("stage2", AND(g["normal"],   cr["elevated"], bu["mild"]))
    fire("stage2", AND(g["normal"],   cr["elevated"], pr["macro"]))
    fire("stage2", AND(g["normal"],   pr["macro"],    bu["mild"]))
    fire("stage2", AND(g["normal"],   cr["elevated"], bp["stage2"]))
    fire("stage2", AND(g["normal"],   su["diabetes"], cr["elevated"],
                       pr["micro"]))

    # Stage 3: high creatinine with normal GFR (muscle wasting / late detection)
    fire("stage3", AND(g["normal"],   cr["high"],     pr["macro"]))
    fire("stage3", AND(g["normal"],   cr["high"],     bu["moderate"]))
    fire("stage3", AND(g["normal"],   pr["macro"],    bu["moderate"]))
    fire("stage3", AND(g["mild"],     cr["high"],     bu["moderate"]))
    fire("stage3", AND(g["mild"],     cr["high"],     pr["macro"]))

    # ─── CATEGORY D: SINGLE CRITICAL BIOMARKER OVERRIDES ─────────────────────
    # Medical basis: any single extreme value signals advanced kidney failure
    # regardless of GFR (which can appear falsely normal in muscle wasting)

    # Creatinine critical (≥8 mg/dL) → Stage 5 — definitive uremia
    fire("stage5", cr["critical"])

    # Creatinine very high (4.5-10) → Stage 4 minimum
    fire("stage4", cr["very_high"])

    # BUN uremic (>115) → Stage 5 — life-threatening
    fire("stage5", bu["uremic"])

    # BUN severe (78-128) → Stage 4 minimum
    # Note: BUN alone is non-specific, so we keep this at stage4 not stage5
    fire("stage4", bu["severe"])

    # Nephrotic proteinuria (>3500 mg/day) → Stage 4 minimum
    fire("stage4", pr["nephrotic"])

    # ─── CATEGORY E: CRITICAL COMBINATION RULES → Stage 4 or 5 ──────────────
    # Medical basis: BUN:Creatinine ratio interpretation (NIH-NCBI NBK305)
    # Two or more critical values together = definitive severe/failure stage

    # Creatinine very_high + any other severe marker → Stage 5
    fire("stage5", AND(cr["very_high"], bu["severe"]))
    fire("stage5", AND(cr["very_high"], bu["uremic"]))
    fire("stage5", AND(cr["very_high"], pr["nephrotic"]))
    fire("stage5", AND(cr["very_high"], bp["crisis"]))
    fire("stage5", AND(cr["very_high"], su["uncontrolled"], bu["moderate"]))

    # Nephrotic + other severe markers → Stage 5
    fire("stage5", AND(pr["nephrotic"], bu["severe"]))
    fire("stage5", AND(pr["nephrotic"], bu["uremic"]))
    fire("stage5", AND(pr["nephrotic"], cr["high"],   bu["moderate"]))

    # BUN uremic + other markers → Stage 5
    fire("stage5", AND(bu["uremic"],    cr["high"]))
    fire("stage5", AND(bu["uremic"],    pr["macro"],  cr["elevated"]))

    # BUN severe + high creatinine + crisis BP → Stage 5
    fire("stage5", AND(bu["severe"],    cr["high"],   bp["crisis"]))
    fire("stage5", AND(bu["severe"],    pr["nephrotic"], bp["stage2"]))

    # Stage 4 combinations (serious but not yet stage5)
    fire("stage4", AND(cr["high"],      bu["severe"]))
    fire("stage4", AND(cr["high"],      bu["moderate"], pr["nephrotic"]))
    fire("stage4", AND(cr["very_high"], pr["macro"]))
    fire("stage4", AND(cr["very_high"], bu["moderate"]))
    fire("stage4", AND(cr["high"],      su["uncontrolled"], bu["moderate"]))
    fire("stage4", AND(bu["severe"],    pr["macro"],   cr["elevated"]))

    return act


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DEFUZZIFICATION
# Hybrid approach:
# 1. Compute standard weighted centroid (center of mass)
# 2. If any critical stage (3/4/5) fires with strength >= 0.5,
#    blend the centroid toward that stage's center.
#    This prevents lower-stage activations from diluting critical overrides.
# ═══════════════════════════════════════════════════════════════════════════════

CENTERS = {
    "healthy": 0.0, "stage1": 1.0, "stage2": 2.0,
    "stage3":  3.0, "stage4": 4.0, "stage5": 5.0,
}
THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]   # stage boundaries

def defuzzify(act):
    # Step 1: standard weighted centroid
    num = sum(act[k] * CENTERS[k] for k in CENTERS)
    den = sum(act[k] for k in CENTERS)
    score = num / den if den else 0.0

    # Step 2: critical stage blend
    # Iterate from most severe to least — stop at first strong activation
    for stage, center in [("stage5", 5.0), ("stage4", 4.0), ("stage3", 3.0)]:
        strength = act[stage]
        if strength >= 0.5:
            score = score * (1.0 - strength) + center * strength
            break

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STAGE METADATA (Labels, risk, recommendations)
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_INFO = [
    (0.5, 0, "طبيعي — لا يوجد مرض كلوي", "Healthy — No CKD", "منخفض", "green",
     ["استمر في نمط الحياة الصحي",
      "أجرِ فحصاً دورياً كل سنة (GFR + بروتين البول)",
      "اشرب 6-8 أكواب ماء يومياً",
      "حافظ على وزن صحي ومارس الرياضة",
      "تجنب NSAIDs (Ibuprofen) وخاصةً إذا كان عندك ضغط أو سكر"]),

    (1.5, 1, "المرحلة الأولى — تلف كلوي مع وظيفة طبيعية",
     "Stage G1 — Kidney Damage with Normal GFR (≥90)", "منخفض إلى متوسط", "yellow",
     ["مراجعة طبيب كل 3 أشهر لمتابعة GFR والبروتين",
      "تحكم في ضغط الدم (الهدف: < 130/80 mmHg)",
      "قلل تناول الصوديوم (< 2.3 غ/يوم) والبروتين الزائد",
      "أوقف التدخين — يُسرّع تدهور وظائف الكلى",
      "تحكم في سكر الدم إن كنت مصاباً بالسكري",
      "تجنب NSAIDs واستخدم Acetaminophen بديلاً"]),

    (2.5, 2, "المرحلة الثانية — انخفاض خفيف في وظيفة الكلى",
     "Stage G2 — Mildly Decreased GFR (60-89)", "متوسط", "orange",
     ["متابعة مع أخصائي كلى كل 3 أشهر",
      "نظام غذائي منخفض البوتاسيوم والفسفور والصوديوم",
      "تحكم صارم في ضغط الدم والسكر",
      "تجنب جميع مضادات الالتهاب (NSAIDs)",
      "راجع طبيبك لتعديل جرعات جميع أدويتك حسب GFR",
      "فحص سنوي للقلب والعظام (مضاعفات CKD)"]),

    (3.5, 3, "المرحلة الثالثة — انخفاض معتدل في وظيفة الكلى",
     "Stage G3 — Moderately Decreased GFR (30-59)", "متوسط إلى عالٍ", "red-light",
     ["إحالة فورية لأخصائي أمراض الكلى (Nephrologist)",
      "نظام غذائي كلوي متخصص مع أخصائي تغذية",
      "مراقبة: البوتاسيوم، الفوسفور، الكالسيوم، الهيموغلوبين",
      "علاج فقر الدم الكلوي إن ظهر (EPO أو الحديد)",
      "تقييم شامل لصحة القلب والأوعية الدموية",
      "بدء التخطيط المبكر للعلاج البديل (dialysis/transplant)"]),

    (4.5, 4, "المرحلة الرابعة — انخفاض حاد في وظيفة الكلى",
     "Stage G4 — Severely Decreased GFR (15-29)", "عالٍ جداً", "red",
     ["إشراف طبي مكثف — زيارات شهرية للنفرولوجيست",
      "التخطيط الجاد لغسيل الكلى (Hemodialysis / Peritoneal)",
      "تقييم مبكر لزراعة الكلى",
      "نظام غذائي صارم: بروتين، بوتاسيوم، فسفور، سوائل",
      "علاج فقر الدم والتحمض الأيضي والاضطرابات المعدنية",
      "مراقبة ضغط الدم يومياً (هدف < 130/80)",
      "تجنب أي نفروتوكسين تماماً"]),

    (99,  5, "المرحلة الخامسة — فشل كلوي (ESRD)",
     "Stage G5 — Kidney Failure / ESRD (GFR < 15)", "حرج", "critical",
     ["⚠️ حالة طارئة — تحتاج تدخلاً طبياً فورياً",
      "بدء غسيل الكلى: Hemodialysis أو Peritoneal Dialysis",
      "تقييم عاجل لزراعة الكلى",
      "علاج مكثف: اليوريميا، التحمض، الاضطرابات الكهربائية",
      "إدارة سوائل صارمة وتغذية كلوية متخصصة",
      "دعم نفسي واجتماعي مكثف للمريض والعائلة",
      "مراجعة وتعديل جميع الأدوية بشكل كامل"]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose(gfr_val, creatinine_val, protein_val, bp_val, sugar_val, bun_val):
    g  = gfr_sets(gfr_val)
    cr = creatinine_sets(creatinine_val)
    pr = protein_sets(protein_val)
    bp = bp_sets(bp_val)
    su = sugar_sets(sugar_val)
    bu = bun_sets(bun_val)

    act   = evaluate_rules(g, cr, pr, bp, su, bu)
    score = defuzzify(act)

    for threshold, stage, label, label_en, risk, color, recs in STAGE_INFO:
        if score < threshold:
            return {
                "score":           round(score, 3),
                "stage":           stage,
                "label":           label,
                "label_en":        label_en,
                "risk":            risk,
                "color":           color,
                "recommendations": recs,
                "activations":     {k: round(v, 3) for k, v in act.items()},
                "memberships": {
                    "gfr":        {k: round(v, 3) for k, v in g.items()},
                    "creatinine": {k: round(v, 3) for k, v in cr.items()},
                    "protein":    {k: round(v, 3) for k, v in pr.items()},
                    "bp":         {k: round(v, 3) for k, v in bp.items()},
                    "sugar":      {k: round(v, 3) for k, v in su.items()},
                    "bun":        {k: round(v, 3) for k, v in bu.items()},
                },
                "inputs": {
                    "gfr": gfr_val, "creatinine": creatinine_val,
                    "protein": protein_val, "bp": bp_val,
                    "sugar": sugar_val, "bun": bun_val,
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COMPREHENSIVE TEST SUITE (20 cases, medically grounded)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # (name, gfr, creatinine, protein, bp, sugar, bun, expected_stage)
    cases = [
        # ── HEALTHY (all normal) ────────────────────────────────────────────
        ("01 كل القيم طبيعية",         105, 0.9,  12, 112,  85, 12,  0),
        ("02 GFR خفيف+كل شيء طبيعي",   78, 1.0,  18, 115,  90, 14,  0),

        # ── STAGE 1: Kidney damage, GFR preserved ──────────────────────────
        ("03 GFR طبيعي + بروتين micro",  95, 1.0,  90, 120,  95, 15,  1),
        ("04 GFR طبيعي + Cr elevated",   98, 1.4,  20, 118,  88, 16,  1),
        ("05 GFR خفيف + بروتين micro",   78, 1.1,  60, 122,  92, 13,  1),

        # ── STAGE 2 ─────────────────────────────────────────────────────────
        ("06 GFR خفيف+Cr+protein",       72, 1.5, 180, 130, 120, 22,  2),
        ("07 GFR طبيعي+Cr+BUN mild",     92, 1.6,  25, 132, 105, 30,  2),
        ("08 GFR خفيف+سكري+Cr",          70, 1.5,  80, 135, 150, 25,  2),

        # ── STAGE 3 ─────────────────────────────────────────────────────────
        ("09 GFR moderate classic",       48, 2.2, 350, 145, 160, 50,  3),
        ("10 GFR خفيف+Cr high+BUN",       65, 2.8, 400, 140, 155, 60,  3),
        ("11 GFR طبيعي+Cr high+BUN mod",  95, 3.0, 500, 142, 150, 65,  3),
        ("12 Cr+protein+BP compound",      80, 2.5, 800, 155, 170, 55,  3),

        # ── STAGE 4 ─────────────────────────────────────────────────────────
        ("13 GFR severe classic",          22, 4.0,1000, 158, 195, 85,  4),
        ("14 Cr very_high override",       85, 7.0, 500, 148, 140, 70,  4),
        ("15 BUN severe override",         88, 2.0, 200, 150, 130,100,  4),
        ("16 Nephrotic proteinuria",       90, 1.8,4000, 145, 140, 80,  4),

        # ── STAGE 5 ─────────────────────────────────────────────────────────
        ("17 GFR طبيعي+Cr+BUN uremic",    97, 1.5,3550, 163,  60,169,  5),
        ("18 GFR failure classic",          8, 9.0,3000, 182, 240,145,  5),
        ("19 Cr critical override",        95,10.0,1000, 155, 180,110,  5),
        ("20 GFR طبيعي+كل القيم حرج",    120,15.0,5000, 200, 400,200,  5),
    ]

    print()
    print("╔" + "═"*78 + "╗")
    print("║  CKD Fuzzy Expert System v3.0 — Test Suite (KDIGO 2024 Based)"
          + " " * 16 + "║")
    print("╠" + "═"*78 + "╣")
    hdr = f"  {'الحالة':28} {'متوقع':8} {'نتيجة':8} {'درجة':7} {'✓':3}"
    print(f"║{hdr:78}║")
    print("╠" + "═"*78 + "╣")

    passed = 0
    for name, gfr, cr, pr, bp, su, bu, exp in cases:
        r  = diagnose(gfr, cr, pr, bp, su, bu)
        ok = r["stage"] == exp
        if ok: passed += 1
        sym = "✅" if ok else "❌"
        row = (f"  {name:28} مرحلة {exp}   مرحلة {r['stage']}"
               f"   {r['score']:<6.3f} {sym}")
        print(f"║{row:78}║")

    print("╠" + "═"*78 + "╣")
    total = len(cases)
    status = ("✅ جميع الحالات صحيحة!" if passed == total
              else f"❌ {passed}/{total} صحيحة")
    print(f"║  النتيجة: {status:<68}║")
    print("╚" + "═"*78 + "╝")