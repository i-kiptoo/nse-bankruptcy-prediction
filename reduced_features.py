import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass 
class FinancialData:
    # Income Statement
    revenue: pd.Series  # Revenue
    ebit: pd.Series # Operating Profit/Loss
    net_income: pd.Series  # Profit/Loss of the Year
    # balance sheet
    current_receivables: pd.Series # Trade And Other Receivables(Current Assets)
    cash_and_equivalents: pd.Series # Cash And Bank Balances
    current_assets: pd.Series # Current Assets
    total_assets: pd.Series # Total Assets
    # liabilities and equity
    current_liabilities: pd.Series #Current Liabilities
    total_liabilities: pd.Series # Total Liabilities
    retained_earnings: pd.Series # Retained Earnings/Accumulated Losses
    # Cash Flow Statement
    net_cash_from_ops: pd.Series  # Net Cash From Operations
    
    def __post_init__(self):
        # Income Statement
        self.revenue = abs(self.revenue)
        self.non_operating_income = self.net_income - self.ebit
        # Balance Sheet
        self.non_current_liabilities = self.total_liabilities - self.current_liabilities
        self.quick_assets = self.cash_and_equivalents + self.current_receivables
    
    def ratios(self):
        df = pd.DataFrame({
            'Non-operating Income Margin': self.non_operating_income / self.revenue,
            'Retained Earnings to Total Assets': self.retained_earnings / self.total_assets,
            'Quick Assets to Total Assets': self.quick_assets / self.total_assets,
            'Operating Cash Flow Ratio': self.net_cash_from_ops / self.current_liabilities,
            'Non-Current Liabilities to Current Assets': self.non_current_liabilities / self.current_assets,
            'Cash and Equivalents to Total Assets': self.cash_and_equivalents / self.total_assets,
            'Liabilities to Asset Ratio': self.total_liabilities / self.total_assets,
            'Current Ratio': self.current_assets / self.current_liabilities,
            'Operating Cash Flow to Revenue': self.net_cash_from_ops / self.revenue,
            'Receivables Turnover Ratio': self.revenue / self.current_receivables,
        })
        
        df = df.replace([np.inf, -np.inf], 0)  # Replace inf and -inf values with 0
        return df