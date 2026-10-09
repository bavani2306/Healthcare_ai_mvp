class MedicalRAG:
    def query(self, question, context_notes=""):
        # Simple mock RAG - later you can connect to ChromaDB / OpenAI
        knowledge = {
            "bp": "High BP causes: stress, obesity, high salt, age. Manage with diet, exercise.",
            "sugar": "High sugar indicates diabetes risk. Maintain low carb diet, monitor regularly.",
            "headache": "Headache can be due to stress, dehydration, BP variation."
        }
        q = question.lower()
        for k, v in knowledge.items():
            if k in q:
                return v
        return f"Based on notes '{context_notes[:100]}...', general advice: Stay hydrated, regular checkup. Q: {question} (This is mock RAG, connect LLM for real)"