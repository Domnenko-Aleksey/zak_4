from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

estimators = [
  ('lr', LinearRegression()),
  ('svr', LinearSVR(random_state=12)),
  ('rfr', RandomForestRegressor(max_depth=2, random_state=12))
]

model = StackingRegressor(
  estimators=estimators,
  final_estimator=RandomForestRegressor(n_estimators=10, random_state=12)
)