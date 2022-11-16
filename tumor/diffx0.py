import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
fig.set_size_inches(8.4, 6)
linewidth = 2
labelsize = 15
figfontsize = 20
bwidth = 4

ax.spines['top'].set_linewidth(bwidth)
ax.spines['left'].set_linewidth(bwidth)
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
ax.tick_params(length=10, width=3, labelsize = labelsize)

ax2 = ax.twinx()
x = np.linspace(0, 0.9999, 1000)
y1 = np.tanh(x)
y2 = np.tanh(x/400)
curve1 = ax.plot(x, y1, color = 'k', marker = '', linestyle=':', linewidth = 5, label=r'$x_0=1$')
curve2 = ax2.plot(x, y2, color = 'r', marker = '', linestyle='-', linewidth = 5, label=r'$x_0=400$')
curves = curve1 + curve2
ax.legend(curves, [curve.get_label() for curve in curves], prop={'size':20}, loc=2)
# ax.legend(prop={'size':20}, loc=4)


xticklabel = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
yticklabel = [0.0, 0.2, 0.4, 0.6]
yticklabel2 = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025]

xtick = xticklabel 
ytick = yticklabel
ax.set_yticks(ytick)
ax.set_xticks(xtick)
ax.set_xticklabels(xticklabel, size=figfontsize)
ax.set_yticklabels(yticklabel, size=figfontsize)
ax2.set_yticklabels(yticklabel2, size=figfontsize)
fig.savefig('./diffx0.svg', format='svg')
plt.show()