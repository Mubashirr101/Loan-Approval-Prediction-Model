import streamlit as st
import pandas as pd
import numpy as np


class LoanApprovalPredApp:
    def __init__(self):
        # title
        st.title("Loan Approval Prediction")
        self.mainpage()

    def mainpage(self):
        st.write("Content")


if __name__ == "__main__":
    LoanApprovalPredApp()
