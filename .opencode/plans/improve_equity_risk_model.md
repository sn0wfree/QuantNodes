# 提升权益风险预警模型

## 目标

当前权益 AUC 0.688 → 目标 ~0.72，修复二十大漏报 (0/10d)。

## 做法

### Phase 1: 宏观因子层（第 5 层）

在 `risk_warning.py` 中新增 `_model_macro()` 函数，使用 VIX/DXY/CN-US spread 滚动回归预测波动率。

**改动文件**: `QuantNodes/strategy/momentum_etf_rotation/v10_3/risk_warning.py`

**步骤**:

1. **添加 HF_DIR 常量** (第 33 行后):
   ```python
   HF_DIR = REPO / "data" / "high_freq_macro"
   ```

2. **新增 `_model_macro()` 函数** (在 `_model_current_vol` 之后):
   ```python
   def _model_macro(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
       from sklearn.linear_model import LinearRegression
       vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
       vix.index = pd.to_datetime(vix.index)
       dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
       dxy.index = pd.to_datetime(dxy.index)
       cs = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
       cs["date"] = pd.to_datetime(cs["date"])
       cs = cs.set_index("date")["cn_us_spread"]
       
       idx = rets.index
       vix = vix.reindex(idx).ffill()
       dxy = dxy.reindex(idx).ffill()
       cs = cs.reindex(idx).ffill()
       
       vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
       dxy_z = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
       cs_z = (cs - cs.rolling(252).mean()) / cs.rolling(252).std()
       
       fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
       forecasts = pd.Series(index=oos_idx, dtype=float)
       refit_dates = oos_idx[::20]
       for rd in refit_dates:
           ri = rets.index.get_indexer([rd], method="ffill")[0]
           if ri < 504:
               continue
           X = pd.DataFrame({"vix_z": vix_z.iloc[:ri], "dxy_z": dxy_z.iloc[:ri],
                             "cs_z": cs_z.iloc[:ri], "c": 1.0})
           y = fwd_vol.iloc[:ri]
           valid = X.notna().all(axis=1) & y.notna()
           Xc, yc = X.loc[valid].values, y.loc[valid].values
           try:
               m = LinearRegression(fit_intercept=False).fit(Xc, yc)
               fs = rd + pd.Timedelta(days=1)
               for fd in pd.bdate_range(fs, periods=20):
                   if fd not in forecasts.index:
                       continue
                   fi = rets.index.get_indexer([fd], method="ffill")[0]
                   feat = np.array([vix_z.iloc[fi-1], dxy_z.iloc[fi-1],
                                    cs_z.iloc[fi-1], 1.0]).reshape(1, -1)
                   if not np.any(np.isnan(feat)):
                       forecasts[fd] = m.predict(feat)[0]
           except Exception:
               continue
       return forecasts.fillna(method="ffill")
   ```

3. **修改 `_fit_all()`** — 在 models dict 中添加 "macro" 键，并在 ensemble 计算后添加 jump overlay:
   ```python
   def _fit_all(self):
       self._raw = {}
       self._prob = {}
       models = {
           "harrv": _model_harrv,
           "climatology": _model_climatology,
           "ewma": _model_ewma,
           "current_vol": _model_current_vol,
           "macro": _model_macro,
       }
       for key, fn in models.items():
           self._raw[key] = fn(self.rets, self.oos_idx)
           common = self._raw[key].dropna().index.intersection(self.y_true.dropna().index)
           self._prob[key] = _cont_prob(self._raw[key].loc[common])
       self._ensemble_prob, self._ensemble_weights = self._compute_ensemble_weights()
       self._ensemble_prob = _jump_overlay(self._ensemble_prob, self.rets)
       self._metrics = {}
   ```

4. **修改 `_compute_ensemble_weights()`** — 将 keys 改为 5 个:
   ```python
   def _compute_ensemble_weights(self, window: int = 60):
       keys = ["harrv", "climatology", "ewma", "current_vol", "macro"]
       ...
   ```

5. **修改 `evaluate()`** — 在 layer 循环中添加 "macro":
   ```python
   for key in ["harrv", "climatology", "ewma", "current_vol", "macro"]:
       ...
   ```

6. **修改 `forecast_df` property** — 添加 macro:
   ```python
   for key in ["harrv", "climatology", "ewma", "current_vol", "macro"]:
       df[f"prob_{key}"] = self._prob[key]
   for i, key in enumerate(["harrv", "climatology", "ewma", "current_vol", "macro"]):
       df[f"weight_{key}"] = self._ensemble_weights.iloc[:, i]
   ```

### Phase 2: 跳跃检测覆盖层

**步骤**:

1. **新增 `_jump_overlay()` 函数** (在 `_model_macro` 之后):
   ```python
   def _jump_overlay(prob: pd.Series, rets: pd.Series,
                      asset_type: str = "equity") -> pd.Series:
       from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
           jump_model_periodic_retrain
       )
       regime = jump_model_periodic_retrain(rets, asset_type=asset_type,
                                             show_progress=False)
       common = prob.index.intersection(regime.index)
       result = prob.loc[common].copy()
       bear_dates = regime.loc[common][regime.loc[common] == 1].index
       if len(bear_dates) > 0:
           result.loc[bear_dates] = np.minimum(result.loc[bear_dates] * 1.3, 1.0)
       return result
   ```

2. **在 `_fit_all()` 中已有调用** (见 Phase 1 步骤 3)

### 验证

```bash
cd /home/ll/Public/QuantNodes && python3 scripts/v10_3/v10_3_risk_backtest.py 2>&1
```

对比前后:

| 指标 | 当前 | 预期 |
|------|------|------|
| AUC | 0.688 | >0.71 |
| 二十大命中 | 0/10d | 改善 |
| 关税冲击命中 | 0/1d | 改善 |
| 滚动 AUC 下限 | 0.0 | 改善 |