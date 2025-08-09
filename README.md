# Statistical Arbitrage Strategy Based on Copulas and GARCH

This strategy aims to detect and exploit temporary misalignments between two financial assets $X$ and $Y$, combining GARCH-based marginal modeling with Gaussian copula dependence modeling. The results of the backtests and analysis are presented in the `main.ipynb` notebook.

---

## 1. Marginal Modeling
For each asset $i \in {X, Y}$, the 5-minute returns $r_{i,t}$ are modeled using a GARCH(1,1) process with Student-\( t \) innovations:

$$
r_{i,t} = \sigma_{i,t} \, \varepsilon_{i,t}, \quad \varepsilon_{i,t} \sim t_{\nu_i}(0, 1)
$$

$$
\sigma_{i,t}^2 = \omega_i + \alpha_i r_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2
$$
  
Standardized residuals are then computed as:

$$
z_{i,t} = \frac{r_{i,t}}{\hat{\sigma}_{i,t}}
$$

---

## 2. Transformation into uniform law
The standardized residuals are transformed into uniform law:

$$
u_{i,t} = F_{t_{\nu_i}}(z_{i,t})
$$

where $F_{t_{\nu_i}}$ is the CDF of the estimated Student-$t$ distribution.

---

## 3. Dependence Modeling
The joint dependence structure is modeled with a Gaussian copula:

$$
C_{\rho}(u_{X,t}, u_{Y,t}) = \Phi_{\rho} \big( \Phi^{-1}(u_{X,t}), \, \Phi^{-1}(u_{Y,t}) \big)
$$

where $\Phi_{\rho}$ is the CDF of a bivariate normal distribution with correlation $\rho$.

---

## 4. Mispricing Index Signal
The conditional probability from the copula is computed as:

$$
P(U_X \leq u_{X,t} \mid U_Y = u_{Y,t}) = \frac{\partial}{\partial u_Y} C_{\rho}(u_{X,t}, u_{Y,t})
$$

**Interpretation:**
- This probability measures how likely it is, given the current state of asset  $Y$, that asset $X$ takes a value less than or equal to its current observed value under the joint dependence structure.
- If the probability is **close to 0.5**, the observation $(u_{X,t}, u_{Y,t})$ lies near the “equilibrium” predicted by the copula — the two assets are fairly priced relative to each other.
- If the probability is **very high** (e.g., $> 0.99$), asset $X$ is in an extreme upper-tail position relative to $Y$, suggesting **potential overvaluation** of $X$ compared to $Y$.
- If the probability is **very low** (e.g., $< 0.01$), asset $X$ is in an extreme lower-tail position relative to $Y$, suggesting **potential undervaluation** of $X$ compared to $Y$.

**Trading rule:**
- If $P > 0.99$ → Short $X$, Long $Y$ (expecting $X$ to revert downward relative to $Y$).
- If $P < 0.01$ → Long $X$, Short  $Y$ (expecting $X$ to revert upward relative to $Y$).

These thresholds (0.99 and 0.01) are chosen to focus only on extreme deviations, reducing noise and avoiding overtrading in equilibrium conditions.
---

## 5. Training and Trading Window
- **Training**: The model is calibrated using data from day $t-1$ (about 78 five-minute observations).  
- **Trading**: The model is applied to day $t$ and recalibrated daily using a rolling window.

---

## 6. Risk Management
When opening a position:
1. Measure the return spread that triggered the signal.  
2. Use this spread as a symmetric take-profit and stop-loss threshold.  
3. Close the position as soon as one of the thresholds is reached.

---



## 7. Future Work: Adjusting for Intraday Volatility Periodicity

High-frequency returns often exhibit a **deterministic intraday volatility pattern**, typically U-shaped — high at the open and close, low in the middle of the day.  
This periodicity can bias both GARCH estimation and dependence modeling in a statistical arbitrage framework.

A general model for the intraday return $R_{t,n}$ (day $t$, intraday interval $n$) can be expressed as:

$$
R_{t,n} = \frac{\sigma_t \, S_n \, Z_{t,n}}{\sqrt{N}}
$$

where:
- $\sigma_t$ = **daily volatility component**, modeled with GARCH(1,1) on daily data,
- $S_n$ = **deterministic intraday periodicity**, estimated using a Fourier flexible form regression,
- $Z_{t,n}$ = i.i.d. standardized innovations with zero mean and unit variance,
- $N$ = number of intraday intervals in a trading day.

### Step 1: Estimating the periodicity $S_n$  
The deterministic intraday volatility pattern $S_n$ can be modeled using Fourier series, which capture the cyclical shape of volatility within a trading day (often U-shaped).


### Step 2: Removing both volatility components  
To isolate pure innovations for the statistical arbitrage model, returns are **double-standardized**:

$$
\tilde{R}_{t,n} = \frac{R_{t,n}}{\hat{\sigma}_t \, \hat{S}_n}
$$


where:
- $\hat{\sigma}_t$ = estimated daily volatility from the GARCH model,
- $\hat{S}_n$ = estimated intraday periodicity from the Fourier regression.

The series $\tilde{R}_{t,n}$ should be closer to an i.i.d. process in both mean and variance, improving the accuracy of copula-based dependence modeling and signal detection.

**Reference:**  
- Andersen, T. G., & Bollerslev, T. (1997). *Intraday Periodicity and Volatility Persistence in Financial Markets*. Journal of Empirical Finance, 4(2–3), 115–158.
