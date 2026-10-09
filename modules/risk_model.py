import pandas as pd

class RiskModel:
    def predict(self, df):
        # Simple rule-based risk for general health
        try:
            row = df.iloc[0]
            score = 0
            if row.get('systolic_bp', 120) > 140: score += 25
            if row.get('diastolic_bp', 80) > 90: score += 15
            if row.get('sugar', 90) > 140: score += 25
            if row.get('heart_rate', 72) > 100: score += 15
            if row.get('bmi', 22) > 28: score += 15
            if row.get('age', 25) > 60: score += 10

            level = "Low" if score < 30 else "Moderate" if score < 60 else "High"
            return {"risk_score": min(score, 95), "level": level, "model": "general_vitals"}
        except Exception as e:
            return {"risk_score": 20, "level": "Low", "error": str(e)}