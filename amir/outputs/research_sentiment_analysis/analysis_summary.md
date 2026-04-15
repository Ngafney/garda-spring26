# Research Sentiment Macro Analysis

## Coverage
- Monthly observations: 45
- Quarterly observations: 16
- Month range: 2022-08-01 to 2026-04-01

## Recent Macro Signal Read
- Latest monthly composite: -0.0005
- Latest monthly growth signal: 0.0099
- Latest monthly pricing-power signal: -0.0000
- Latest monthly macro-risk density: 0.0011

## Strongest Univariate Fits By R-squared
- ew_supply_chain_pressure_density -> cpi_yoy_3m_fwd: coef=0.821, t=6.96, p=0.000, R^2=0.674, n=40
- ew_supply_chain_pressure_density -> core_cpi_yoy_3m_fwd: coef=0.787, t=7.73, p=0.000, R^2=0.620, n=40
- ew_margin_signal -> core_cpi_yoy_3m_fwd: coef=-0.760, t=-5.30, p=0.000, R^2=0.577, n=40
- ew_margin_signal -> cpi_yoy_3m_fwd: coef=-0.685, t=-4.01, p=0.000, R^2=0.469, n=40
- ew_margin_signal -> cpi_yoy_change_3m_fwd: coef=0.613, t=4.39, p=0.000, R^2=0.375, n=39
- ew_supply_chain_pressure_density -> cpi_yoy_change_3m_fwd: coef=-0.589, t=-7.59, p=0.000, R^2=0.347, n=39
- ew_labor_pressure_density -> cpi_yoy_3m_fwd: coef=0.544, t=2.46, p=0.014, R^2=0.296, n=40
- ew_labor_pressure_density -> core_cpi_yoy_3m_fwd: coef=0.428, t=1.99, p=0.046, R^2=0.184, n=40
- ew_growth_signal -> core_cpi_yoy_3m_fwd: coef=-0.428, t=-1.76, p=0.078, R^2=0.183, n=40
- ew_supply_chain_pressure_density -> core_cpi_yoy_change_3m_fwd: coef=-0.391, t=-2.62, p=0.009, R^2=0.153, n=39

## Strongest Lead/Lag Correlations
- Labor Pressure -> Unemployment at lead_+6m: corr=-0.589
- Growth -> Unemployment at lead_+6m: corr=0.504
- Growth -> Unemployment at lead_-3m: corr=0.444
- Growth -> Unemployment at lead_+1m: corr=0.442
- Labor Pressure -> Unemployment at lead_+4m: corr=-0.424
- Labor Pressure -> Unemployment at lead_+5m: corr=-0.418
- Growth -> Unemployment at lead_+0m: corr=0.378
- Growth -> Unemployment at lead_+2m: corr=0.371
- Growth -> Unemployment at lead_-2m: corr=0.363
- Growth -> Unemployment at lead_+4m: corr=0.357

## Multivariate Monthly Models
- cpi_yoy_change_3m_fwd:
  ew_supply_chain_pressure_density: coef=-0.627, t=-3.79, p=0.000
  ew_uncertainty_density: coef=0.567, t=1.54, p=0.123
  ew_pricing_power_net: coef=-0.441, t=-2.26, p=0.024
  ew_growth_signal: coef=0.269, t=1.48, p=0.138
  ew_margin_signal: coef=0.241, t=1.18, p=0.237
- core_cpi_yoy_change_3m_fwd:
  ew_supply_chain_pressure_density: coef=-0.551, t=-2.67, p=0.008
  ew_uncertainty_density: coef=0.462, t=1.29, p=0.197
  ew_labor_pressure_density: coef=-0.215, t=-0.78, p=0.436
  ew_pricing_power_net: coef=-0.142, t=-0.46, p=0.643
  ew_macro_risk_density: coef=0.126, t=0.39, p=0.695
- unemployment_change_3m_fwd:
  ew_supply_chain_pressure_density: coef=-0.158, t=-0.79, p=0.431
  fed_growth_score: coef=-0.138, t=-0.54, p=0.591
  share_guidance_raised: coef=0.093, t=0.36, p=0.720
  ew_labor_pressure_density: coef=-0.090, t=-0.38, p=0.706
  fed_inflation_score: coef=0.060, t=0.28, p=0.779