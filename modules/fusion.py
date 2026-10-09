class FusionEngine:
    def fuse(self, vitals_risk, vision_risk, rag_result):
        v_score = vitals_risk.get('risk_score', 0)
        img_score = vision_risk.get('risk_score', 0)

        # Weighted fusion: 60% vitals, 30% image, 10% notes
        final_risk = (v_score * 0.6) + (img_score * 0.3) + 5

        if final_risk < 30:
            level = "Low Risk"
            rec = "Maintain healthy lifestyle, regular monitoring."
        elif final_risk < 65:
            level = "Moderate Risk"
            rec = "Consult general physician, follow up tests recommended."
        else:
            level = "High Risk"
            rec = "Immediate medical consultation recommended."

        return {
            "final_risk": round(final_risk, 2),
            "final_level": level,
            "recommendation": rec,
            "components": {
                "vitals": vitals_risk,
                "vision": vision_risk
            }
        }