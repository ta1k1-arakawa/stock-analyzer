import pandas as pd
from src.analysis import create_target_variable
from src.trade_simulator import PortfolioSettings, simulate_execution, simulate_portfolio


def frame(rows):
    return pd.DataFrame(rows, index=pd.date_range("2025-01-01", periods=len(rows), freq="B"))


class TestTradeSimulator:
 def test_normal_stop_uses_stop_price(self):
  df=frame([{"Open":100,"Low":99,"Close":100},{"Open":100,"Low":94,"Close":96}])
  x=simulate_execution(df,0,1,5,stop_slippage_percent=1)
  assert x.exit_reason == "STOP" and x.exit_price == 94.05


 def test_gap_stop_uses_open_not_stop_price(self):
  df=frame([{"Open":100,"Low":99,"Close":100},{"Open":90,"Low":89,"Close":90}])
  x=simulate_execution(df,0,1,5,stop_slippage_percent=1)
  assert x.exit_price == 89.1


 def test_label_matches_execution_costs(self):
  df=frame([{"Open":100,"Low":99,"Close":100},{"Open":100,"Low":94,"Close":96}])
  labelled=create_target_variable(df,1,-5,0,0,5,1,0)
  assert labelled.iloc[0].Target == 0


 def test_exit_cash_cannot_fund_same_day_open(self):
  signals=[
      {"code":"A","prob":.9,"entry_date":"2025-01-02","exit_date":"2025-01-03","entry_price":100,"exit_price":110},
      {"code":"B","prob":.8,"entry_date":"2025-01-03","exit_date":"2025-01-06","entry_price":100,"exit_price":110},]
  trades, ledger=simulate_portfolio(signals,100,PortfolioSettings(max_open_positions=1))
  assert trades[0]["status"] == "FILLED" and trades[1]["status"] != "FILLED"
  assert all(x["cash"] >= 0 for x in ledger)
