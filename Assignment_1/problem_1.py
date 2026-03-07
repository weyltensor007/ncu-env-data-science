import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# generate x points by linspace and logsapce
n_points = 1000
x_linspace = np.linspace(0.01, 10, n_points)
x_logspace = np.logspace(-3, 2, n_points)

# set figure layout
fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

# set xlim, ylim for different plots
linear_lim = (0, 10)
log_lim = (1e-3, 1e2)

# set colors for different functions
color_x = "tab:orange"
color_exp = "tab:blue"
color_log = "tab:green"

# set major tick for linear and logarithmic
ticks_linear = [0, 2, 4, 6, 8, 10]
ticks_logarithmic = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

'''
upper left x:linear, y:linear
'''
ax = axes[0, 0]  # fix ax
# plot different functions
ax.plot(x_linspace, x_linspace, color=color_x)
ax.plot(x_linspace, 10**(x_linspace), color=color_exp)
ax.plot(x_linspace, np.log10(x_linspace), color=color_log)
# set x,y limits on plot
ax.set_xlim(*linear_lim)
ax.set_ylim(*linear_lim)
# set y major ticks (no x ticks for this ax)
ax.set_yticks(ticks_linear)
# set x,y minor ticks every 0.4(need minor ticks for grid lines)
ax.xaxis.set_minor_locator(MultipleLocator(0.4))
ax.yaxis.set_minor_locator(MultipleLocator(0.4))
# set major and minor grid lines
ax.grid(True, which='major', linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', linestyle='-', alpha=0.5)
# hide x ticks
ax.tick_params(axis='x',
               which='both',
               bottom=False,  # hide bottom ticks
               top=False,  # hide top ticks
               labelbottom=False  # hide numbers on x-axis
               )
# label functions on curves
ax.text(
    5.6,
    4.8,
    r"$f(x)=x$",
    color=color_x,
    fontsize=14,
    ha='left',
    va='bottom')
ax.text(
    1.2,
    8.4,
    r"$f(x)=10^{x}$",
    color=color_exp,
    fontsize=14,
    ha='left',
    va='bottom'
)
ax.text(
    5.6,
    1.2,
    r"$f(x)=\log_{10}(x)$",
    color=color_log,
    fontsize=14,
    ha='left',
    va='bottom'
)
# set title and y-label
ax.set_title("X linear - Y linear", fontsize=16)
ax.set_ylabel("Linear", fontsize=16)


'''
upper right x:log, y:linear
'''
ax = axes[0, 1]  # fix ax
ax.set_xscale('log')  # set x axis to be log scale

# plot different functions
ax.plot(x_logspace, x_logspace, color=color_x)
ax.plot(x_logspace, 10**(x_logspace), color=color_exp)
ax.plot(x_logspace, np.log10(x_logspace), color=color_log)

# set x,y limits on plot
ax.set_xlim(*log_lim)
ax.set_ylim(*linear_lim)

# set x-axis ticks at 10^-3, 10^-2, ..., 10^2
ax.set_xticks(ticks_logarithmic)
# hide x ticks
ax.tick_params(axis='x',
               which='both',
               bottom=False,  # hide bottom ticks
               top=False,  # hide top ticks
               labelbottom=False  # hide numbers on x-axis
               )

# set y minor ticks every 0.4
ax.yaxis.set_minor_locator(MultipleLocator(0.4))
# hide y ticks
ax.tick_params(
    axis='y',       # apply to y-axis
    which='both',   # major and minor ticks
    left=False,     # hide left ticks
    right=False,    # hide right ticks
    labelleft=False  # hide numbers on y-axis
)
# show grid lines
ax.grid(True, which='major', linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', linestyle='-', alpha=0.5)
# set title
ax.set_title("X logarithmic - Y linear", fontsize=16)

'''
lower left x:linear, y:log
'''
ax = axes[1, 0]
ax.set_yscale("log")
ax.set_xlim(*linear_lim)
ax.set_ylim(*log_lim)
ax.plot(x_linspace, x_linspace, color=color_x)
ax.plot(x_linspace, 10**x_linspace, color=color_exp)
ax.plot(x_linspace, np.log10(x_linspace), color=color_log)
ax.xaxis.set_minor_locator(MultipleLocator(0.4))
ax.grid(True, which='major', linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', linestyle='-', alpha=0.5)
ax.set_title("X linear - Y logarithmic", fontsize=16)
ax.set_ylabel("Logarithmic", fontsize=16)
ax.set_xlabel("Linear", fontsize=16)
'''
lower right x:log, y:log
'''
ax = axes[1, 1]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*log_lim)
ax.set_ylim(*log_lim)
ax.plot(x_logspace, x_logspace, color=color_x)
ax.plot(x_logspace, 10**x_logspace, color=color_exp)
ax.plot(x_logspace, np.log10(x_logspace), color=color_log)
ax.grid(True, which='major', linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', linestyle='-', alpha=0.5)
ax.set_title("X logarithmic - Y logarithmic", fontsize=16)
ax.set_xlabel("Logarithmic", fontsize=16)

ax.tick_params(
    axis='y',       # apply to y-axis
    which='both',   # major and minor ticks
    left=False,     # hide left ticks
    right=False,    # hide right ticks
    labelleft=False  # hide numbers on y-axis
)
# plt.show()
plt.savefig("Assignment_1_problem1.png", dpi=600)
