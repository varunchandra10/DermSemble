import os
import json
from backend.schemas.response_schema import ClassProbability
from backend.inference.model_loader import model_loader
from backend.core.logger import log

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "clinical_knowledge_corpus.json")


class ReportService:
    """
    Generates structured, doctor-grade clinical summaries and triage recommendations
    powered by an in-house curated 25-class clinical knowledge base, calibrated
    acuity multipliers, and selective uncertainty abstention thresholds.
    """
    def __init__(self):
        self.corpus = {}
        self.load_corpus()

    def load_corpus(self):
        try:
            if os.path.exists(CORPUS_PATH):
                with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                    self.corpus = json.load(f)
                log.info(f"[ReportService] Successfully loaded clinical corpus with {len(self.corpus)} classes.")
            else:
                log.warning(f"[ReportService] Corpus file not found at {CORPUS_PATH}. Using fallback templates.")
        except Exception as e:
            log.error(f"[ReportService] Error loading clinical knowledge corpus: {e}")

    def get_clinical_monograph(self, class_name: str) -> dict | None:
        """Retrieves raw structured medical monograph for a specific disease class."""
        return self.corpus.get(class_name)

    def generate_clinical_summary(
        self,
        top_predictions: list[ClassProbability],
        is_abstained: bool = False,
        clinical_tier: str = "Tier 2"
    ) -> str:
        """
        Synthesizes an exhaustive, structured clinical summary covering:
        1. Primary Triage Impression
        2. Uncertainty Abstention Status
        3. Differential Diagnostic Stratification
        4. Etiology & Pathophysiology
        5. Morphology Across Skin Phototypes (Fitzpatrick I-VI)
        6. Confirmatory Diagnostic Workup
        7. Frontline Management & Red-Flag Contraindications
        """
        if not top_predictions:
            return "No predictions available. Please upload a clear dermatological image."

        top_pred = top_predictions[0]
        raw_class = top_pred.class_name
        conf_pct = top_pred.probability * 100.0
        
        entry = self.corpus.get(raw_class, {})
        display_name = entry.get("clinical_name", raw_class.replace("_", " "))
        tier = clinical_tier or entry.get("tier", model_loader.get_tier(raw_class))
        icd10 = entry.get("icd10", "Unspecified")

        sections = []

        # =====================================================================
        # 1. PRIMARY DIAGNOSTIC IMPRESSION
        # =====================================================================
        sections.append(
            f"CLINICAL DIAGNOSTIC IMPRESSION:\n"
            f"• Primary Suspected Condition : {display_name} (ICD-10: {icd10})\n"
            f"• Calibrated AI Confidence     : {conf_pct:.1f}%\n"
            f"• Clinical Acuity Tier         : {tier} ({entry.get('urgency', 'Standard Triage')})"
        )

        # =====================================================================
        # 2. UNCERTAINTY ABSTENTION ALERT (If Triggered)
        # =====================================================================
        if is_abstained:
            sections.append(
                "⚠️ CLINICAL UNCERTAINTY ALERT (SYSTEM ABSTENTION TRIGGERED):\n"
                f"The ensemble prediction confidence ({conf_pct:.1f}%) is below the safety threshold (θ = 50.0%). "
                "Per clinical decision-theoretic safety protocol, automated triage is withheld. "
                "This lesion presents ambiguous visual boundaries. An immediate in-person dermatoscopic "
                "evaluation and tissue biopsy are mandatory to prevent diagnostic delay."
            )

        # =====================================================================
        # 3. DIFFERENTIAL DIAGNOSTIC STRATIFICATION
        # =====================================================================
        if len(top_predictions) > 1:
            diff_lines = []
            for idx, p in enumerate(top_predictions[1:4], start=2):
                diff_entry = self.corpus.get(p.class_name, {})
                d_name = diff_entry.get("clinical_name", p.class_name.replace("_", " "))
                diff_lines.append(f"  {idx}. {d_name} — {p.probability * 100:.1f}%")
            sections.append("DIFFERENTIAL DIAGNOSTIC CONSIDERATIONS:\n" + "\n".join(diff_lines))

        # =====================================================================
        # 4. PATHOPHYSIOLOGY & ETIOLOGY
        # =====================================================================
        if "pathophysiology" in entry:
            sections.append(f"PATHOPHYSIOLOGY & ETIOLOGY:\n{entry['pathophysiology']}")

        # =====================================================================
        # 5. MORPHOLOGICAL HALLMARKS ACROSS FITZPATRICK PHOTOTYPES
        # =====================================================================
        if "morphology" in entry:
            m = entry["morphology"]
            morph_text = (
                f"DERMATOLOGICAL MORPHOLOGY:\n"
                f"• Key Clinical Hallmarks : {m.get('hallmarks', '')}\n"
                f"• Light Skin (FST I–III)  : {m.get('fitzpatrick_light', '')}\n"
                f"• Dark Skin (FST IV–VI)   : {m.get('fitzpatrick_dark', '')}"
            )
            sections.append(morph_text)

        # =====================================================================
        # 6. CONFIRMATORY DIAGNOSTIC WORKUP
        # =====================================================================
        if "diagnostic_workup" in entry and entry["diagnostic_workup"]:
            workup_lines = [f"  • {step}" for step in entry["diagnostic_workup"]]
            sections.append("RECOMMENDED DIAGNOSTIC WORKUP:\n" + "\n".join(workup_lines))

        # =====================================================================
        # 7. FRONTLINE MANAGEMENT
        # =====================================================================
        if "management" in entry:
            sections.append(f"FRONTLINE MANAGEMENT GUIDELINES:\n{entry['management']}")

        # =====================================================================
        # 8. CLINICAL RED FLAGS & CONTRAINDICATIONS
        # =====================================================================
        red_flag_text = entry.get("red_flags", None)
        if not red_flag_text:
            if tier == "Tier 1":
                red_flag_text = "Time-sensitive pathology. Rule out rapid local extension and systemic invasion."
            else:
                red_flag_text = "Correlate with patient symptom duration and rule out atypical refractory presentations."
        
        sections.append(f"RISK NOTE:\n{red_flag_text}")

        return "\n\n".join(sections)


report_service = ReportService()
