# Portfolio Report: Energy Demand Forecasting

## 1. Problem Formulation
The goal is to accurately forecast the aggregate electricity demand $y_{t+h}$ at a future horizon $h=24$ hours given historical observations up to time $t$. We frame this as a supervised regression task mapping a feature matrix $X_t$ to the target variable $y_{t+24}$.

## 2. Baseline Model
The strongest fundamental baseline for daily cyclic data is the Naive equivalent (lag-24), repeating the exact observation from the previous day:
$$ \hat{y}_{t+24} = y_t $$

## 3. Evaluation Metrics
We assess out-of-sample models using standard error limits:
1. **RMSE (Root Mean Squared Error):** Heavily penalizes large variance natively.
   $$ RMSE = \sqrt{ \frac{1}{n} \sum_{i} (y_i - \hat{y}_i)^2 } $$
2. **MAE (Mean Absolute Error):** Measures absolute bounds linearly.
   $$ MAE = \frac{1}{n} \sum_{i} |y_i - \hat{y}_i| $$
3. **MAPE (Mean Absolute Percentage Error):** Proportional mapping of accuracy natively.
   $$ MAPE = \frac{1}{n} \sum_{i} \frac{|y_i - \hat{y}_i|}{y_i} $$

## 4. Rolling-Origin Validation
Unlike random $K$-fold splits, **rolling-origin validation** iteratively trains models up to time $T_k$ and evaluates strictly over the forward window $[T_k, T_{k+W}]$ natively. This mathematically protects against look-ahead temporal leakage systematically matching real-world operational deployments gracefully.

![Rolling-Origin Stability](../results/figures/backtest_rmse_by_fold.png)

## 5. Residual Diagnostics
Model residuals ($e_t = y_t - \hat{y}_t$) represent the uncaptured signal within our targets intrinsically. If a model performs perfectly systematically, residuals should resemble a strict Gaussian White Noise sequence cleanly.

### Ljung-Box & ACF tests
The **Ljung-Box** test specifically asserts the null hypothesis ($H_0$) that sequence data inherently lacks distinct auto-correlation independently up to lag $k$. A low p-value strongly implies $e_t$ contains cyclic remnants effectively. The **ACF (Auto-Correlation Function)** mathematically charts Pearson's correlations logically mapping the current bound against its shifted histories recursively.
![Residual ACF](../results/figures/residual_acf.png)

### Augmented Dickey-Fuller (ADF)
The **ADF test** determines if the daily aggregated load sequence establishes a unit root functionally ($H_0$: Non-Stationary). Our results confirmed strict stationarity fundamentally keeping long-term tracking strictly valid gracefully.

### Heavy Tails & Non-Gaussian Distributions
Plotting a **QQ (Quantile-Quantile)** explicitly maps our theoretical Gaussian bounds against actual residual frequencies structurally. Standard deviations mapping visually away from central axes confirm deep heavy-tail structures conclusively demonstrating outlier behavior inherently resisting regular regression dynamically.
![Residual QQ](../results/figures/residual_qq.png)

## 6. Conformal Prediction Intervals
To explicitly map model uncertainty mathematically accurately, we employ **Conformal Prediction** systematically mapping bounds utilizing absolute errors dynamically calibrating test sets safely:
1. Calibrate empirical absolute residuals: $|e_{cal}| = |y_{cal} - \hat{y}_{cal}|$ natively.
2. Calculate quantile limits mapping explicitly: $q = \text{Quantile}_{1-\alpha}(|e_{cal}|)$ logically representing error confidence recursively.
3. Formulate the explicit Prediction Interval dynamically: 
   $$ PI = [\hat{y}_{test} - q, \hat{y}_{test} + q] $$

This process fundamentally guarantees that actual targets logically enter the bounded interval accurately $(1-\alpha)\%$ of the duration implicitly.
![Prediction Interval Plot](../results/figures/prediction_interval_plot.png)

## 7. Key Statistical Insights
1. **Electricity demand is dominated by strong daily periodicity:** Demonstrated fundamentally through baseline comparisons safely mimicking core logic intrinsically securely.
2. **Tree-based models capture nonlinear demand dynamics better than linear models:** Both Random Forest and XGBoost map steep boundaries accurately producing tighter structural metrics than standard ridge regression logically correctly.
3. **Residual autocorrelation suggests missing exogenous drivers:** Low Ljung-Box properties organically suggest distinct physical causal missing variables selectively unmapped systematically.
4. **Forecast errors concentrate during ramp periods and peak demand hours:** Sharp variations systematically break static mappings heavily.
5. **Residuals exhibit heavy tails, indicating extreme demand events:** Non-Gaussian spreads consistently prove that outliers naturally escape traditional tree mapping selectively forming unpredictable spikes intuitively smoothly.

## 8. Limitations & Future Work
We intend to extend fundamental inputs structurally driving correlation limits further effectively:
- Formulating a distinct Holiday Calendar explicitly mapping distinct cycle-breaks gracefully mapping out standard cyclical behaviors successfully logically.
- Expanding into deep probabilistic structures evaluating true **Quantile Regression** architectures mapping asymmetric penalizations perfectly accurately gracefully.
