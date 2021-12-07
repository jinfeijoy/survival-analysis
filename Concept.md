# Concept

* **Survival Function *`S(t) = Pr(T>t)`***: the probability that the time of death is later than some specified time t.
* **Survival Curve**: a flat survival curve (i.e. one that stays close to 1.0) suggests very good survival, whereas a survival curve that drops sharply toward 0 suggests poor survival.
* **Lifetime distribution function and event density**: 
  * Lifetime distribution function: <img src="https://render.githubusercontent.com/render/math?math=F(t) = Pr(T<=t) = 1 - S(t)">
  * Event density: <img src="https://render.githubusercontent.com/render/math?math=f(t) = F^'(t) = \frac{d}{dt}F(t)">, which indicate the rate of death or failure events per unit time (overall probability density of failing at time t)
  * Survival function: <img src="https://render.githubusercontent.com/render/math?math=S(t) = Pr(T>t) = 1 - F(t) = \int_t^{inf} f(u)du">
  * Survival event density: <img src="https://render.githubusercontent.com/render/math?math=s(t) = S^'(t) = \frac{d}{dt}S(t)=\frac{d}{dt}\int_t^{inf}=\frac{d}{dt}[1-F(t)]=-f(t)">
* **Hazard**
  * Hazard function: donated as <img src="https://render.githubusercontent.com/render/math?math=\lambda"> or *h*, is defined as the event rate at time *t* conditional on survival until time *t* or later (i.e. *T>=t*).
    * Given survived for a time *t* and will not survival for an additional time *dt*: <img src="https://render.githubusercontent.com/render/math?math=h(t)=\lim_{dt\rightarrow 0}\frac{Pr(t\le T < t%2Bdt )}{dt*S(t)}=\frac{f(t)}{S(t)} = - \frac{S^'(t)}{S(t)} = -\frac{d}{dx}ln(S(x))">
    * The hazard function must be non-negative, and its integral over `[0, Inf]` must be infinite, but is not otherwise constrained; it may be increasing or decreasing, non-monotonic or discontinuous.
  * Cumulative hazard function: denoted as `H`
    * `H(t) = -log S(t)` or `S(t) = exp(-H(t))`
    * <img src="https://render.githubusercontent.com/render/math?math=S(t)=exp[-H(t)]=\frac{f(t)}{\lambda(t)}=1-F(t), t>0">
    * Cumulative hazard function measures the total amount of risk that has been accumulated up to time t.
    * Cumulative hazard is like the total number of revolutions an automobile's engine makes over a given period.
* **Life expectancy**: life expectancy can be expressed as an integral of the survival curve
  * Probability of death at or before age `t0 + t` given survival until age `t0` is:<img src="https://render.githubusercontent.com/render/math?math=P(T\le t_0 %2B t|T>t_0)= \frac{P(t_0<T\le T_0%2Bt)}{P(T>t_0)}=\frac{F(t_0%2Bt)-F(t_0)}{S(t_0)}"> 
  * The probability density of future lifetime is: <img src="https://render.githubusercontent.com/render/math?math=\frac{d}{dt}\frac{F(t_0 %2B t) - F(t_0)}{S(t_0)}=\frac{f(t_0%2Bt)}{S(t_0)}">
  * The expected future lifetime: <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{S(t_0)}\int_0^{\infty}tf(t_0%2Bt)dt=\frac{1}{S(t_0)}\int_{t_0}^{\infty}S(t)dt">
* **Censoring**:
  * Censoring is a form of missing data problem in which time to event is not observed for reasons such as termination of study before all recruited subjects have shown the event of interest or the subject has left the study prior to experiencing an event.
  * Right Censored: for those subjects whose birthdate is known but who are still alive when they are lost to follow-up or when the study ends
  * Left Censored: If the event of interest has already happened before the subject is included in the study but it is not known when it occurred
  * Interval censored: the event happened between two observations or examinations
* **Truncation (delayed entry study)**:
  * In a so-called delayed entry study, subjects are not observed at all until they have reached a certain age. For example, people may not be observed until they have reached the age to enter school. Any deceased subjects in the pre-school age group would be unknown. 
  * Left truncated: Left truncation occurs when individuals who have already passed the milestone at the time of study recruitment are not included in the study.

# Survival Analysis
* **Non-parametric Method**
  * Kaplan-Meier Plot: A plot of the Kaplan–Meier estimator is a series of declining horizontal steps which, with a large enough sample size, approaches the true survival function for that population. 
    * As a non-parametric estimator, it does a good job of giving a quick look at the survival curve for a dataset. However, what it doesn’t let you do is model the impact of covariates on survival.
  * Life table: In actuarial science and demography, a life table (also called a mortality table or actuarial table) is a table which shows, for each age, what the probability is that a person of that age will die before their next birthday ("probability of death"). In other words, it presents the survivorship of people from a certain population.
  * Log-rank test(Mantel-Cox test): The log-rank test is a hypothesis test to compare the survival distribution of two samples. It is a non-parametric test and appropriate to use when the data are right skewed and censored.
  * Nelson Aalen Fitter: Like the Kaplan-Meier Fitter, Nelson Aalen Fitter also gives us an average view of the population. It is given by the number of deaths at time t divided by the number of subjects at risk.
* **Parametric Method**: strong assumptions about the data.
  * Weibull Distribution: denoted <img src="https://render.githubusercontent.com/render/math?math=W(p,\lambda),p>0 (shape),\lambda >0 (scale)">
    * Cumulative Distribution Function: <img src="https://render.githubusercontent.com/render/math?math=F(t) = 1 - e^{-(\lambda t)^p}">
    * <img src="https://render.githubusercontent.com/render/math?math=f(t) = p\lambda ^pt^{p-1}e^{-(\lambda t)^p}">
    * <img src="https://render.githubusercontent.com/render/math?math=S(t) = e^{-(\lambda t)^p}">
    * <img src="https://render.githubusercontent.com/render/math?math=h(t) = p\lambda ^pt^{p-1}">
    * <img src="https://render.githubusercontent.com/render/math?math=H(t) = (\lambda t)^p">, `p>1` hazard function is increasing while `p<1`hazard function is decreasing
    * <img src="https://render.githubusercontent.com/render/math?math=W(1,\lambda) = Exp(\lambda)">
    * mean time between failures (expected lifetime): <img src="https://render.githubusercontent.com/render/math?math=MTBF(k,\lambda) = \frac{1}{\lambda}\Gamma (1 %2B \frac{1}{p})"> where <img src="https://render.githubusercontent.com/render/math?math=\Gamma(\alpha)=\int_0^{\infty}t^{\alpha -1}e^{-t}dt">
  * Log-logistic Distribution: 
    * <img src="https://render.githubusercontent.com/render/math?math=F(t)=\frac{x^\beta}{\alpha ^{\beta} %2B x^{\beta}}, scale:\alpha, shape:\beta">
    * <img src="https://render.githubusercontent.com/render/math?math=f(t)=\frac{(\beta / \alpha)(x/ \alpha)^{\beta - 1}}{(1%2B(x/\alpha)^{\beta})^2}">
    * <img src="https://render.githubusercontent.com/render/math?math=S(t)=1-F(t)=[1%2B(t/\alpha)^{\beta}]^{-1}">
    * <img src="https://render.githubusercontent.com/render/math?math=h(t)=\frac{f(t)}{S(t)}=\frac{(\beta/\alpha)(t/\alpha)^{\beta-1}}{1%2B(t/\alpha)^{\beta}}">
    * <img src="https://render.githubusercontent.com/render/math?math=E(T) = \frac{\pi \alpha \beta^{-1}}{\sin(\pi\beta^{-1})}, \beta>1">
    * sometimes <img src="https://render.githubusercontent.com/render/math?math=\mu=ln(\alpha)"> and <img src="https://render.githubusercontent.com/render/math?math=s=1/\beta">, <img src="https://render.githubusercontent.com/render/math?math=\mu,s">in analogy with the logistic distribution
  * Exponential Distribution: denoted <img src="https://render.githubusercontent.com/render/math?math=T~Exp(\lambda)">
    * <img src="https://render.githubusercontent.com/render/math?math=f(t) = \lambda \exp ^{-\lambda t}"> for <img src="https://render.githubusercontent.com/render/math?math=\lambda>0"> (scale parameter)
    * <img src="https://render.githubusercontent.com/render/math?math=F(t) = 1 - \exp ^{-\lambda t}">  
    * <img src="https://render.githubusercontent.com/render/math?math=S(t) = \exp ^{-\lambda t}"> 
    * <img src="https://render.githubusercontent.com/render/math?math=h(t)=\lambda">  constant hazard function
    * <img src="https://render.githubusercontent.com/render/math?math=H(t) = \lambda t">
    * <img src="https://render.githubusercontent.com/render/math?math=E(T) = \frac{1}{\lambda}">
  * Accelerated Failure Time Regression Model
* **Semi-parametric Method**: Proportional hazards model, no functional assumptions are made about the shape of the Hazard Function
  * Cox Proportional-Hazards Model
    * When we are trying to model the effects of covariates (e.g. age, gender, race, machine manufacturer) we will typically be interested in understanding the effect of the covariate on the Hazard Rate. The hazard rate is the instantaneous probability of failure/death/state transition at a given time t, conditional on already having survived that long.
    * The Cox Proportional Hazards Model is usually given in terms of the time t, covariate vector x, and coefficient vector β as <img src="https://render.githubusercontent.com/render/math?math=\lambda(t)=\lambda_0(t)e^{x^T\beta}">, where <img src="https://render.githubusercontent.com/render/math?math=\lambda_0"> s an arbitrary function of time, the baseline hazard. The dot product of X and β is taken in the exponent just like in standard linear regression.
    * The survivor function: <img src="https://render.githubusercontent.com/render/math?math=S(t)=\exp\int_0^t \lambda_0(t)\exp(x^T\beta) = S_0(t)^{\exp(x^T\beta)}">
    * Baseline survivor function: estimating the survivor function is very similar to estimating a KM curve.
    * [Use Cox models only if interested in hazard ratios and nothing else](https://stats.stackexchange.com/questions/68737/how-to-estimate-baseline-hazard-function-in-cox-model-with-r)
    * Concern: keep in mind "omitted variable bias", baseline hazard function is non-parametric, only baseline hazard rate provided with R/Python packages, then the baseline hazard rate will be used in the model.
  * [Aalen’s Additive Model](http://www.ukm.my/jsm/pdf_files/SM-PDF-46-3-2017/15%20Aditif%20Aalen.pdf)
    * <img src="https://render.githubusercontent.com/render/math?math=\lambda(t|X) = \lambda_0(t)r(X^T\beta)"> where <img src="https://render.githubusercontent.com/render/math?math=\lambda_0(t)"> is the baseline hazard and it may have a specified parametric form or may be left as an arbitrary non-negative function.
* **Tree-structured survival models**
  * survival tree analysis
  * survival random forest
* **Performance Evaluation**
  * [Concordance Index(C-index)](https://medium.com/analytics-vidhya/concordance-index-72298c11eac7)
    * [Compute the C-index](https://statisticaloddsandends.wordpress.com/2019/10/26/what-is-harrells-c-index/): for every pair of patients i and j with <img src="https://render.githubusercontent.com/render/math?math=i\ne j">, look at their risk scores and times-to-event.
    * Harrell's C-index is simply: <img src="https://render.githubusercontent.com/render/math?math=c=\frac{# concordant pairs}{# concordant pairs %2B discordant pairs}">

### Reference
* https://en.wikipedia.org/wiki/Survival_analysis#Example:_Acute_myelogenous_leukemia_survival_data
* https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Survival/BS704_Survival_print.html
* https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
* https://www.scielo.br/j/aabc/a/JnPjpjqLKnNzbkDHSQdHx9t/?format=pdf&lang=en
* https://medium.com/codex/survival-analysis-part-ii-ddbbae048d3f
* [Kaggle Example](https://www.kaggle.com/taimurzahid/survival-regression-analysis-to-predict-churn)
* Cox Proportional Hazard Model
  * https://kowshikchilamkurthy.medium.com/the-cox-proportional-hazards-model-da61616e2e50
  * https://medium.com/analytics-vidhya/predict-survival-model-using-cox-proportional-hazard-model-7bb4ee9fec9a
  * https://towardsdatascience.com/the-cox-proportional-hazards-model-35e60e554d8f
  * https://towardsdatascience.com/survival-analysis-part-a-70213df21c2e