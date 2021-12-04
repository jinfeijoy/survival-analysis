# Concept

* **Survival Function *`S(t) = Pr(T>t)`***: the probability that the time of death is later than some specified time t.
* **Survival Curve**: a flat survival curve (i.e. one that stays close to 1.0) suggests very good survival, whereas a survival curve that drops sharply toward 0 suggests poor survival.
* **Lifetime distribution function and event density**: 
  * Lifetime distribution function: <img src="https://render.githubusercontent.com/render/math?math=F(t) = Pr(T<=t) = 1 - S(t)">
  * Event density: <img src="https://render.githubusercontent.com/render/math?math=f(t) = F^'(t) = \frac{d}{dt}F(t)">, which indicate the rate of death or failure events per unit time
  * Survival function: <img src="https://render.githubusercontent.com/render/math?math=S(t) = Pr(T>t) = 1 - F(t) = \int_t^{inf} f(u)du">
  * Survival event density: <img src="https://render.githubusercontent.com/render/math?math=s(t) = S^'(t) = \frac{d}{dt}S(t)=\frac{d}{dt}\int_t^{inf}=\frac{d}{dt}[1-F(t)]=-f(t)">
* **Hazard**
  * Hazard function: donated as <img src="https://render.githubusercontent.com/render/math?math=\lambda"> or *h*, is defined as the event rate at time *t* conditional on survival until time *t* or later (i.e. *T>=t*).
    * Given survived for a time *t* and will not survival for an additional time *dt*: <img src="https://render.githubusercontent.com/render/math?math=h(t)=\lim_{dt\rightarrow 0}\frac{Pr(t-dt\le T -dt < t )}{dt*S(t)}=\frac{f(t)}{S(t)} = - \frac{S^'(t)}{S(t)}">
    * The hazard function must be non-negative, and its integral over `[0, Inf]` must be infinite, but is not otherwise constrained; it may be increasing or decreasing, non-monotonic or discontinuous.
  * Cumulative hazard function: denoted as `H`
    * `H(t) = -log S(t)` or `S(t) = exp(-H(t))`
    * <img src="https://render.githubusercontent.com/render/math?math=S(t)=exp[-H(t)]=\frac{f(t)}{\lambda(t)}=1-F(t), t>0">
    * Cumulative hazard function measures the total amount of risk that has been accumulated up to time t.
    * Cumulative hazard is like the total number of revolutions an automobile's engine makes over a given period.
* **Life expectancy**: life expectancy can be expressed as an integral of the survival curve
  * 
* **Censoring**:
  * Censoring is a form of missing data problem in which time to event is not observed for reasons such as termination of study before all recruited subjects have shown the event of interest or the subject has left the study prior to experiencing an event.
  * Right Censored: for those subjects whose birthdate is known but who are still alive when they are lost to follow-up or when the study ends
  * Left Censored: If the event of interest has already happened before the subject is included in the study but it is not known when it occurred
  * Interval censored: the event happened between two observations or examinations
* **Truncation (delayed entry study)**:
  * In a so-called delayed entry study, subjects are not observed at all until they have reached a certain age. For example, people may not be observed until they have reached the age to enter school. Any deceased subjects in the pre-school age group would be unknown. 
  * Left truncated: Left truncation occurs when individuals who have already passed the milestone at the time of study recruitment are not included in the study.


### Reference
* https://en.wikipedia.org/wiki/Survival_analysis#Example:_Acute_myelogenous_leukemia_survival_data
* https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Survival/BS704_Survival_print.html
