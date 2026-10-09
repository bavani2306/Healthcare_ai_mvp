import streamlit as st

class SafetyChecker:
    def show_disclaimer(self):
        st.warning("⚠️ DISCLAIMER: This is an educational MVP. Not a medical diagnosis tool. Always consult a qualified doctor.")

    def is_safe(self, text):
        unsafe = ["suicide", "self harm"]
        for w in unsafe:
            if w in text.lower():
                return False
        return True