import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


def calculate_sample_size(alpha, beta, mu1, mu2, sigma, tail='two'):
    # Z value
    if tail == 'two':
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)

    # effect size
    delta = abs(mu1 - mu2)

    # sample size
    n = ((z_alpha + z_beta)**2 * (2 * sigma**2)) / (delta**2)

    return math.ceil(n) # round up


# 1. Parameter Settings
mu0 = 0           # H0 Mean
mu1 = 0.03        # H1 Mean (Effect Size)
sigma_pop = 1     # standard deviation of population
alpha = 0.05      # Significance level
beta = 0.2        # 1 - Power

n_required = calculate_sample_size(alpha, beta, mu0, mu1, sigma_pop)
print(f"sample size: {n_required}")
sigma = sigma_pop * np.sqrt(2 / n_required)

# 2. Statistical Points
# Critical Value (H0, alpha/2, two-tailed test)
z_critical = norm.ppf(1 - alpha/2, mu0, sigma)

actual_beta = norm.cdf(z_critical, mu1, sigma)
print(f"beta: {actual_beta:.6f}")

actual_alpha = norm.cdf(z_critical, mu0, sigma)
print(f"alpha: {1-actual_alpha:.6f}")

# 3. Prepare Data
x = np.linspace(mu0 - 4.6*sigma, mu1 + 4*sigma, 1000)
y0 = norm.pdf(x, mu0, sigma)
y1 = norm.pdf(x, mu1, sigma)

# 4. Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the H0 and H1 normal distributions
plt.plot(x, y0, 'k-', lw=2, label='$H_0$')
plt.plot(x, y1, 'k-', lw=2, label='$H_1$')

plt.text(mu0, max(y1)*0.77, '$H_0$', fontsize=14, ha='center', fontweight='bold')
plt.text(mu1, max(y1)*0.77, '$H_1$', fontsize=14, ha='center', fontweight='bold')

# the critical value threshold
plt.axvline(z_critical, color='gray', linestyle='--', lw=1.5)

# -- Shaded Area --

# False Positive (alpha) - under H0, right to z
x_alpha = np.linspace(z_critical, x[-1], 100)
plt.fill_between(x_alpha, norm.pdf(x_alpha, mu0, sigma), color='red', alpha=0.3, hatch='//', label='False Positive (α)')
x_alpha = np.linspace(x[0], 0-z_critical, 100)
plt.fill_between(x_alpha, norm.pdf(x_alpha, mu0, sigma), color='red', alpha=0.3, hatch='//',)

# False Negative (beta) - under H1, left to z
x_beta = np.linspace(x[0], z_critical, 100)
plt.fill_between(x_beta, norm.pdf(x_beta, mu1, sigma), color='blue', alpha=0.2,
                 hatch='', label='False Negative (β)')

# -- Annotations --

# FP and FN arrows
plt.annotate(f'P(False Positive)\n' + fr'$\alpha = {alpha}$',
             xy=(z_critical + sigma*0.19, max(y1)*0.02),
             xytext=(z_critical + sigma*1.2, max(y1)*0.12),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=10)

plt.annotate('P(False Negative)\n 1-Power = 20%',
             xy=(z_critical - sigma*0.47, max(y1)*0.08),
             xytext=(z_critical - sigma*2.8, max(y1)*0.28),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=10)

# Variance / Sample Size Arrow
y_pos = max(y1) * 0.38
dist = sigma * np.sqrt(-2 * np.log(y_pos * sigma * np.sqrt(2 * np.pi)))
x_right = mu1 + dist*0.97

plt.annotate('', xy=(mu1, y_pos), xytext=(x_right, y_pos),
             arrowprops=dict(arrowstyle='<->', lw=0.5),
             color='#555555')

plt.text((mu1 + x_right)/2, y_pos,
         r'$\propto 1/\sqrt{n}$',
         ha='center', va='bottom', fontweight='bold',)

# Reject / Fail to Reject Arrows
y_pos = max(y1) * 1.02
dist = sigma*0.5

# Reject H0
plt.annotate('', xy=(z_critical+0.001, y_pos), xytext=(z_critical+dist, y_pos),
             arrowprops=dict(arrowstyle='<-', lw=0.5),
             color='#555555')
plt.text(z_critical+dist, y_pos,
         r'$Reject\ H_0$',
         ha='left', #va='bottom',
         fontweight='bold',)

# Fail to reject H0
plt.annotate('', xy=(z_critical-0.001, y_pos), xytext=(z_critical-dist, y_pos),
             arrowprops=dict(arrowstyle='<-', lw=0.5),
             color='#555555')
plt.text(z_critical-dist, y_pos,
         r'$Fail\ to\ reject\ H_0$',
         ha='right', #va='bottom',
         fontweight='bold',)

# axes format
plt.yticks([])  # Hide Y-axis values
plt.xticks([0, mu1], ['0', r'Effect Size'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# -- Decision Matrix Table --
table_data = [['Correct\n' + fr'Power, $1-\beta$', f'False Negative\n' + fr'$\beta$'],
              [f'False Positive\n' + fr'$\alpha$', 'Correct\n' + fr'$1-\alpha$']]
row_labels = [r'$H_1$ is True', r'$H_0$ is True']
col_labels = [r'Reject $H_0$', r'Fail to reject $H_0$']

the_table = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='upper left',
    cellLoc='center',
    bbox=[0.05, 0.7, 0.28, 0.22]
)

# cell colors
cells = the_table.get_celld()
cells[(1, 0)].set_facecolor('#e6ffed')       # Correct Power
cells[(2, 0)].set_facecolor((1, 0, 0, 0.3))  # False Negative (Type II)
cells[(1, 1)].set_facecolor((0, 0, 1, 0.2))  # False Positive (Type I)

# Remove label borders
for (row, col), cell in the_table.get_celld().items():
    cell.set_edgecolor('gray')
    if row == 0 or col == -1:
        cell.set_linewidth(0)
        cell.set_facecolor('none')

the_table.auto_set_font_size(False)
the_table.set_fontsize(10)

# plt.savefig("power_analysis_plot.svg", format="svg", bbox_inches='tight')
plt.show()


