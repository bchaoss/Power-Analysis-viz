# Power-Analysis-viz
A visualization of the relationships among the five elements in experimental power analysis:

**Effect Size**, **Sample Size ($n$)**, **Significance Level ($\alpha$)**, **Power ($1-\beta$)**, and **Standard Deviation ($\sigma$)**. 

![power_analysis_viz](viz/power_analysis_viz.svg)

It's important to determine the required power level and smaple size before running an A/B testing.

Inspired by several excellent power analysis tools / sample size calculators, I generated this chart to show the trade-off relationships between the key metrics in power analysis and how we could increase statistical power:

| How to | Visualization | Note |
|--|--|--|
| Increase effect size | Larger distance between $H_0$ and $H_1$ | Stronger "signal" is easier to detect |
| Derease SD ($\sigma$) | Narrower distributions | Reduce noise, ask clear questions |
| Increase sample size ($n$) | Narrower distributions | More data points shrink the standard error |
| Change test type ($z -> t$) | Modifing the shape of distributions | Choice depends on the underlying data/assumptions |
| Increase significance level ($\alpha$) | Move the reject line left | Dangerous, could be wrong  |

**In visual eyes, statistical power is determined by: the relative positions of the $H_0$ and $H_1$ distributions, and the placement of the decision criteria ($\alpha$) relative to those distributions.**

Methods 1–4 shift the former by reducing distribution overlap. The "dangerous" 5th affects the latter by simply lowering the rejection bar. Among these, increasing sample size ($n$) is usually the most practical approach.

---

The main code drawing the chart is in `scripts/plot.py`, and the future plan is a web-based chart for interactive exploration.

Reference:

1. https://clincalc.com/stats/samplesize.aspx
2. https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
3. https://towardsdatascience.com/5-ways-to-increase-statistical-power-377c00dd0214/
