import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb


class LoanApprovalPredApp:
    def __init__(self):
        # title
        st.title("Loan Approval Prediction")
        self.mainpage()

    def mainpage(self):
        st.write("Content")
        xgb_model = xgb.Booster()
        xgb_model.load_model("src/xgb_model.json")

        # input
        st.write("modelmade")


if __name__ == "__main__":
    LoanApprovalPredApp()
