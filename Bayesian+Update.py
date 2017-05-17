
# coding: utf-8

# In[4]:

# graph examples from matplotlib example page

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)


fig, axScatter = plt.subplots(figsize=(5.5, 5.5))

# the scatter plot:
axScatter.scatter(x, y)
axScatter.set_aspect(1.)

# create new axes on the right and on the top of the current axes
# The first argument of the new_vertical(new_horizontal) method is
# the height (width) of the axes to be created in inches.
divider = make_axes_locatable(axScatter)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

# make some labels invisible
plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
         visible=False)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
lim = (int(xymax/binwidth) + 1)*binwidth

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
for tl in axHistx.get_xticklabels():
    tl.set_visible(False)
axHistx.set_yticks([0, 50, 100])

#axHisty.axis["left"].major_ticklabels.set_visible(False)
for tl in axHisty.get_yticklabels():
    tl.set_visible(False)
axHisty.set_xticks([0, 50, 100])

plt.draw()
plt.show()


# In[19]:

outlook = ['s','s','o','r','r','r','o','s','s','r','s','o','o','r']
temperature = ['h','h','h','m','c','c','c','m','c','m','m','m','h','m']
humidity = ['h','h','h','h','n','n','n','h','n','n','n','h','n','h']
#windy = [False,True,False,False,False,True,True,False,False,False,True,True,False,True]
windy = ['f','t','f','f','f','t','t','f','f','f','t','t','f','t']
play = ['n','n','y','y','y','n','y','n','y','y','y','y','y','n']

attribute = [outlook, temperature, humidity, windy, play]

newex = ['s', 'c', 'h', 't'] # a new day

#print outlook.count(newex[0])
#print play.count(newex[4])

def countif(mylist, val):
    return len([x for x in mylist if x == val])

def combctif(lista, listb, vala, valb):
    count = 0
    ind = 0
    for x in lista:
        #print x, listb[ind]
        if x == vala and listb[ind] == valb:
            count += 1
        ind += 1
    return count

# calculate probabilities from frequencies
# later need to adjust: Laplace Estimator

cy = float(countif(play, 'y'))
cn = float(countif(play, 'n'))

# frequencies for the new day
co = float(combctif(attribute[0], attribute[4], newex[0], 'y'))
ct = float(combctif(attribute[1], attribute[4], newex[1], 'y'))
ch = float(combctif(attribute[2], attribute[4], newex[2], 'y'))
cw = float(combctif(attribute[3], attribute[4], newex[3], 'y'))
cp = float(combctif(attribute[4], attribute[4], 'y', 'y'))

likelihoody = co / cy * ct / cy * ch / cy * cw / cy * cp / (cy + cn)
print 'likelihood yes: %f' % likelihoody

co = float(combctif(attribute[0], attribute[4], newex[0], 'n'))
ct = float(combctif(attribute[1], attribute[4], newex[1], 'n'))
ch = float(combctif(attribute[2], attribute[4], newex[2], 'n'))
cw = float(combctif(attribute[3], attribute[4], newex[3], 'n'))
cp = float(combctif(attribute[4], attribute[4], 'n', 'n'))

likelihoodn = co / cn * ct / cn * ch / cn * cw / cn * cp / (cy + cn)
print 'likelihood no: %f' % likelihoodn

# P(yes) = likel y / (likel y + likel n) = P(y|E)P(E)/(P(y|E)P(E) + P(y|E)P(E))
print 'prob yes: %f' % (likelihoody / (likelihoody + likelihoodn))
print 'prob no: %f' % (likelihoodn / (likelihoody + likelihoodn))

# change play so that it is always n if outlook = sunny:
play = ['n','n','y','y','y','n','y','n','n','y','n','y','y','n']

cy = float(countif(play, 'y'))
cn = float(countif(play, 'n'))

my = 0.2 # Laplace estimator

co = float(combctif(attribute[0], attribute[4], newex[0], 'y') + my/3)
ct = float(combctif(attribute[1], attribute[4], newex[1], 'y'))
ch = float(combctif(attribute[2], attribute[4], newex[2], 'y'))
cw = float(combctif(attribute[3], attribute[4], newex[3], 'y'))
cp = float(combctif(attribute[4], attribute[4], 'y', 'y'))

likelihoody = co / (cy + my) * ct / cy * ch / cy * cw / cy * cp / (cy + cn)
print 'likelihood yes: %f' % likelihoody

co = float(combctif(attribute[0], attribute[4], newex[0], 'n'))
ct = float(combctif(attribute[1], attribute[4], newex[1], 'n'))
ch = float(combctif(attribute[2], attribute[4], newex[2], 'n'))
cw = float(combctif(attribute[3], attribute[4], newex[3], 'n'))
cp = float(combctif(attribute[4], attribute[4], 'n', 'n'))

likelihoodn = co / cn * ct / cn * ch / cn * cw / cn * cp / (cy + cn)
print 'likelihood no: %f' % likelihoodn

# P(yes) = likel y / (likel y + likel n) = P(y|E)P(E)/(P(y|E)P(E) + P(y|E)P(E))
print 'prob yes: %f' % (likelihoody / (likelihoody + likelihoodn))
print 'prob no: %f' % (likelihoodn / (likelihoody + likelihoodn))


# In[8]:

outlook = ['s','s','o','r','r','r','o','s','s','r','s','o','o','r']
temperature = ['h','h','h','m','c','c','c','m','c','m','m','m','h','m']
humidity = ['h','n','h','h','h','n','n','h','n','n','n','h','n','h']
#windy = [False,True,False,False,False,True,True,False,False,False,True,True,False,True]
windy = ['f','t','f','f','f','t','t','f','f','f','t','t','f','t']
#play = ['n','n','y','y','y','n','y','n','y','y','y','y','y','n']
# change play so that it is always n if outlook = sunny:
play = ['n','n','y','y','y','n','y','n','n','y','n','y','y','n']

attribute = [outlook, temperature, humidity, windy, play]

newex = ['s', 'c', 'h', 't'] # a new day

#print outlook.count(newex[0])
#print play.count(newex[4])

def countif(mylist, val):
    return len([x for x in mylist if x == val])

def combctif(lista, listb, vala, valb):
    count = 0
    ind = 0
    for x in lista:
        #print x, listb[ind]
        if x == vala and listb[ind] == valb:
            count += 1
        ind += 1
    return count

cy = float(countif(play, 'y'))
cn = float(countif(play, 'n'))

my = 0.2 # Laplace estimator
# needed if any P(Ei|yes)=0
mydiv = max(len(set(outlook)),1) # getting unique items from list: set(outlook)
# priors for outlook - assume equal probabilities
p1 = 0.3
p2 = 0.3
p3 = 0.3

zerofreqchk = [] # any P=0? P <- likelihood <- frequencies: check if frequency=0
#co = float(combctif(attribute[0], attribute[4], newex[0], 'y') + my * p1)
co = float(combctif(attribute[0], attribute[4], newex[0], 'y'))
zerofreqchk.append(co)
ct = float(combctif(attribute[1], attribute[4], newex[1], 'y'))
zerofreqchk.append(ct)
ch = float(combctif(attribute[2], attribute[4], newex[2], 'y'))
zerofreqchk.append(ch)
cw = float(combctif(attribute[3], attribute[4], newex[3], 'y'))
zerofreqchk.append(cw)
cp = float(combctif(attribute[4], attribute[4], 'y', 'y'))
zerofreqchk.append(cp)

#print 0 in zerofreqchk # frequency=0
#print zerofreqchk.index(0)
#if 0 in zerofreqchk:
#    zerofreqchk[zerofreqchk.index(0)] += my * p1

likelihoody = 1.0
for x in zerofreqchk:
    if x == 0:
        likelihoody *= (x + my * p1) / ( cy + my )
    else:
        likelihoody *= x / cy

#likelihoody = co / (cy + my) * ct / cy * ch / cy * cw / cy * cp / (cy + cn)
print 'likelihood yes: %f' % likelihoody

co = float(combctif(attribute[0], attribute[4], newex[0], 'n'))
ct = float(combctif(attribute[1], attribute[4], newex[1], 'n'))
ch = float(combctif(attribute[2], attribute[4], newex[2], 'n'))
cw = float(combctif(attribute[3], attribute[4], newex[3], 'n'))
cp = float(combctif(attribute[4], attribute[4], 'n', 'n'))

likelihoodn = co / cn * ct / cn * ch / cn * cw / cn * cp / (cy + cn)
print 'likelihood no: %f' % likelihoodn

# P(yes) = likel y / (likel y + likel n) = P(y|E)P(E)/(P(y|E)P(E) + P(y|E)P(E))
print 'prob yes: %f' % (likelihoody / (likelihoody + likelihoodn))
print 'prob no: %f' % (likelihoodn / (likelihoody + likelihoodn))


# In[5]:

import matplotlib.pyplot as plt
from numpy.random import rand


fig, ax = plt.subplots()
for color in ['red', 'green', 'blue']:
    n = 750
    x, y = rand(2, n)
    scale = 200.0 * rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()


# In[6]:

'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[7]:

'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()


# In[8]:

"""
Demonstrates using custom hillshading in a 3D surface plot.
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

filename = cbook.get_sample_data('jacksboro_fault_dem.npz', asfileobj=False)
with np.load(filename) as dem:
    z = dem['elevation']
    nrows, ncols = z.shape
    x = np.linspace(dem['xmin'], dem['xmax'], ncols)
    y = np.linspace(dem['ymin'], dem['ymax'], nrows)
    x, y = np.meshgrid(x, y)

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()


# In[11]:

'''
==================================================
A simple example of a quiver plot with a quiverkey
==================================================
'''
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

plt.show()


# In[12]:

"""
Demo of a function to create Hinton diagrams.

Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
a weight matrix): Positive and negative values are represented by white and
black squares, respectively, and the size of each square represents the
magnitude of each value.

Initial idea from David Warde-Farley on the SciPy Cookbook
"""
import numpy as np
import matplotlib.pyplot as plt


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


if __name__ == '__main__':
    hinton(np.random.rand(20, 20) - 0.5)
    plt.show()


# In[13]:

"""
================
The Bayes update
================

This animation displays the posterior estimate updates as it is refitted when
new data arrives.
The vertical line represents the theoretical value to which the plotted
distribution should converge.
"""

# update a distribution based on new data.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.animation import FuncAnimation


class UpdateDist(object):
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = ss.beta.pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

fig, ax = plt.subplots()
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, ud, frames=np.arange(100), init_func=ud.init,
                     interval=100, blit=True)
plt.show()


# In[14]:

"""
Demo of the `streamplot` function.

A streamplot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the stream plot function:

    * Varying the color along a streamline.
    * Varying the density of streamlines.
    * Varying the line width along a stream line.
"""
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

fig1, (ax1, ax2) = plt.subplots(ncols=2)
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

lw = 5*speed / speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()


# In[15]:

"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


# In[16]:

"""
========================================
Bayesian Methods for Hackers style sheet
========================================

This example demonstrates the style used in the Bayesian Methods for Hackers
[1]_ online book.

.. [1] http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/

"""
from numpy.random import beta
import matplotlib.pyplot as plt


plt.style.use('bmh')


def plot_beta_hist(ax, a, b):
    ax.hist(beta(a, b, size=10000), histtype="stepfilled",
            bins=25, alpha=0.8, normed=True)


fig, ax = plt.subplots()
plot_beta_hist(ax, 10, 10)
plot_beta_hist(ax, 4, 12)
plot_beta_hist(ax, 50, 12)
plot_beta_hist(ax, 6, 55)
ax.set_title("'bmh' style sheet")

plt.show()


# In[17]:

"""
================
The Sankey class
================

Demonstrate the Sankey class by producing three basic diagrams.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.sankey import Sankey


# Example 1 -- Mostly defaults
# This demonstrates how to create a simple diagram by implicitly calling the
# Sankey.add() method and by appending finish() to the call to the class.
Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
       labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
       orientations=[-1, 1, 0, 1, 1, 1, 0, -1]).finish()
plt.title("The default settings produce a diagram like this.")
# Notice:
#   1. Axes weren't provided when Sankey() was instantiated, so they were
#      created automatically.
#   2. The scale argument wasn't necessary since the data was already
#      normalized.
#   3. By default, the lengths of the paths are justified.

# Example 2
# This demonstrates:
#   1. Setting one path longer than the others
#   2. Placing a label in the middle of the diagram
#   3. Using the scale argument to normalize the flows
#   4. Implicitly passing keyword arguments to PathPatch()
#   5. Changing the angle of the arrow heads
#   6. Changing the offset between the tips of the paths and their labels
#   7. Formatting the numbers in the path labels and the associated unit
#   8. Changing the appearance of the patch and the labels after the figure is
#      created
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Flow Diagram of a Widget")
sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
                format='%.0f', unit='%')
sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
           labels=['', '', '', 'First', 'Second', 'Third', 'Fourth',
                   'Fifth', 'Hurray!'],
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
           pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
                        0.25],
           patchlabel="Widget\nA")  # Arguments to matplotlib.patches.PathPatch()
diagrams = sankey.finish()
diagrams[0].texts[-1].set_color('r')
diagrams[0].text.set_fontweight('bold')
# Notice:
#   1. Since the sum of the flows is nonzero, the width of the trunk isn't
#      uniform.  If verbose.level is helpful (in matplotlibrc), a message is
#      given in the terminal window.
#   2. The second flow doesn't appear because its value is zero.  Again, if
#      verbose.level is helpful, a message is given in the terminal window.

# Example 3
# This demonstrates:
#   1. Connecting two systems
#   2. Turning off the labels of the quantities
#   3. Adding a legend
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
sankey = Sankey(ax=ax, unit=None)
sankey.add(flows=flows, label='one',
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
sankey.add(flows=[-0.25, 0.15, 0.1], label='two',
           orientations=[-1, -1, -1], prior=0, connect=(0, 0))
diagrams = sankey.finish()
diagrams[-1].patch.set_hatch('/')
plt.legend(loc='best')
# Notice that only one connection is specified, but the systems form a
# circuit since: (1) the lengths of the paths are justified and (2) the
# orientation and ordering of the flows is mirrored.

plt.show()


# In[18]:

import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
from matplotlib.cbook import get_sample_data

fname = get_sample_data('percent_bachelors_degrees_women_usa.csv')
gender_degree_data = csv2rec(fname)

# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# You typically want your plot to be ~1.33x wider than tall. This plot
# is a rare exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 14))

# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
ax.set_xlim(1969.5, 2011.1)
ax.set_ylim(-0.25, 90)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(range(1970, 2011, 10), fontsize=14)
plt.yticks(range(0, 91, 10), fontsize=14)
ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted.
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='on', left='off', right='off', labelleft='on')

# Now that the plot is prepared, it's time to actually plot the data!
# Note that I plotted the majors in order of the highest % in the final year.
majors = ['Health Professions', 'Public Administration', 'Education',
          'Psychology', 'Foreign Languages', 'English',
          'Communications\nand Journalism', 'Art and Performance', 'Biology',
          'Agriculture', 'Social Sciences and History', 'Business',
          'Math and Statistics', 'Architecture', 'Physical Sciences',
          'Computer Science', 'Engineering']

y_offsets = {'Foreign Languages': 0.5, 'English': -0.5,
             'Communications\nand Journalism': 0.75,
             'Art and Performance': -0.25, 'Agriculture': 1.25,
             'Social Sciences and History': 0.25, 'Business': -0.75,
             'Math and Statistics': 0.75, 'Architecture': -0.75,
             'Computer Science': 0.75, 'Engineering': -0.25}

for rank, column in enumerate(majors):
    # Plot each line separately with its own color.
    column_rec_name = column.replace('\n', '_').replace(' ', '_').lower()

    line = plt.plot(gender_degree_data.year,
                    gender_degree_data[column_rec_name],
                    lw=2.5,
                    color=color_sequence[rank])

    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    y_pos = gender_degree_data[column_rec_name][-1] - 0.5

    if column in y_offsets:
        y_pos += y_offsets[column]

    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    plt.text(2011.5, y_pos, column, fontsize=14, color=color_sequence[rank])

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
fig.suptitle('Percentage of Bachelor\'s degrees conferred to women in '
             'the U.S.A. by major (1970-2011)\n', fontsize=18, ha='center')

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')
plt.show()


# In[19]:

"""
====
XKCD
====

Shows how to create an xkcd-like plot.
"""
import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Monroe
    # http://xkcd.com/418/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    ax.set_ylim([-30, 10])

    data = np.ones(100)
    data[70:] -= np.arange(30)

    plt.annotate(
        'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

    plt.plot(data)

    plt.xlabel('time')
    plt.ylabel('my overall health')
    fig.text(
        0.5, 0.05,
        '"Stove Ownership" from xkcd by Randall Monroe',
        ha='center')

    # Based on "The Data So Far" from XKCD by Randall Monroe
    # http://xkcd.com/373/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.bar([0, 1], [0, 100], 0.25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([0, 1])
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([0, 110])
    ax.set_xticklabels(['CONFIRMED BY\nEXPERIMENT', 'REFUTED BY\nEXPERIMENT'])
    plt.yticks([])

    plt.title("CLAIMS OF SUPERNATURAL POWERS")

    fig.text(
        0.5, 0.05,
        '"The Data So Far" from xkcd by Randall Monroe',
        ha='center')

plt.show()


# In[2]:

# from https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

"""
The book uses a custom matplotlibrc file, which provides the unique styles for
matplotlib plots. If executing this book, and you wish to use the book's
styling, provided are two options:
    1. Overwrite your own matplotlibrc file with the rc-file provided in the
       book's styles/ dir. See http://matplotlib.org/users/customizing.html
    2. Also in the styles is  bmh_matplotlibrc.json file. This can be used to
       update the styles in only this notebook. Try running the following code:

        import json
        s = json.load(open("../styles/bmh_matplotlibrc.json"))
        matplotlib.rcParams.update(s)

"""

# The code below can be passed over, as it is currently not important, plus it
# uses advanced topics we have not covered yet. LOOK AT PICTURE, MICHAEL!
get_ipython().magic(u'matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)

import scipy.stats as stats

dist = stats.beta
n_trials = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)

# For the already prepared, I'm using Binomial's conj. prior.
for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials)/2, 2, k+1)
    plt.xlabel("$p$, probability of heads")         if k in [0, len(n_trials)-1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)

    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)


plt.suptitle("Bayesian updating of posterior probabilities",
             y=1.02,
             fontsize=14)

plt.tight_layout()


# In[21]:

x = range(10)
print x[:3]


# In[22]:

figsize(12.5, 4)
p = np.linspace(0, 1, 50)
plt.plot(p, 2*p/(1+p), color="#348ABD", lw=3)
#plt.fill_between(p, 2*p/(1+p), alpha=.5, facecolor=["#A60628"])
plt.scatter(0.2, 2*(0.2)/1.2, s=140, c="#348ABD")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Prior, $P(A) = p$")
plt.ylabel("Posterior, $P(A|X)$, with $P(A) = p$")
plt.title("Are there bugs in my code?");


# In[23]:

figsize(12.5, 4)
colours = ["#348ABD", "#A60628"]

prior = [0.20, 0.80]
posterior = [1./3, 2./3]
plt.bar([0, .7], prior, alpha=0.70, width=0.25,
        color=colours[0], label="prior distribution",
        lw="3", edgecolor=colours[0])

plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
        width=0.25, color=colours[1],
        label="posterior distribution",
        lw="3", edgecolor=colours[1])

plt.xticks([0.20, .95], ["Bugs Absent", "Bugs Present"])
plt.title("Prior and Posterior probability of bugs present")
plt.ylabel("Probability")
plt.legend(loc="upper left");


# In[3]:

figsize(12.5, 4)

import scipy.stats as stats
a = np.arange(16)
poi = stats.poisson
lambda_ = [1.5, 4.25]
colours = ["#348ABD", "#A60628"]

plt.bar(a, poi.pmf(a, lambda_[0]), color=colours[0],
        label="$\lambda = %.1f$" % lambda_[0], alpha=0.60,
        edgecolor=colours[0], lw="3")

plt.bar(a, poi.pmf(a, lambda_[1]), color=colours[1],
        label="$\lambda = %.1f$" % lambda_[1], alpha=0.60,
        edgecolor=colours[1], lw="3")

plt.xticks(a + 0.4, a)
plt.legend()
plt.ylabel("probability of $k$")
plt.xlabel("$k$")
plt.title("Probability mass function of a Poisson random variable; differing $\lambda$ values");


# In[7]:

a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.8, 0.4]

for l, c in zip(lambda_, colours):
    plt.plot(a, expo.pdf(a, scale=1./l), lw=3,
             color=c, label="$\lambda = %.1f$" % l)
    plt.fill_between(a, expo.pdf(a, scale=1./l), color=c, alpha=.33)

plt.legend()
plt.ylabel("PDF at $z$")
plt.xlabel("$z$")
plt.ylim(0,1.2)
plt.title("Probability density function of an Exponential random variable; differing $\lambda$");


# In[22]:

# naive Bayes model
# objective: classify new examples for which the class variable C is unobserved based on observed attributes x1,...,xn
# probability of each class: P(C|x1,...,xn)=aP(C) TT P(xi|C)

# theta: true shape (solid, empty, diagonal, horizontal, vertica)
# theta0-3: observed shape: 2x2 pixels: x0 to x3

import numpy as np
from matplotlib import pyplot as plt

# generate random 4-sequence of pixels
# x0 x1
# x2 x3

def fourpix():
    return np.random.randint(2, size=(1,4))
#print fourpix()

# identify shape
def whichshape(s):
    if (s == [[1,1,1,1]]).all():
        return 1 # solid
    elif (s == [[0,0,0,0]]).all():
        return 2 # empty
    elif (s == [[0,1,0,1]]).all():
        return 3 # vertical
    elif (s == [[1,0,1,0]]).all():
        return 3 # vertical
    elif (s == [[1,1,0,0]]).all():
        return 4 # horizontal
    elif (s == [[0,0,1,1]]).all():
        return 4 # horizontal
    elif (s == [[1,0,0,1]]).all():
        return 5 # diagonal
    elif (s == [[0,1,1,0]]).all():
        return 5 # diagonal
    else:
        return 0
#print whichshape(fourpix())

def ssd(A,B):
    s = 0
    for i in range(5):
        s += (A[i] - B[i]) * (A[i] - B[i])
    return s

# add counter to respective shape counter: solid, empty, vertical, horizontal, diagonal
# result should converge towards [0.06 0.06 0.125 0.125 0.125]
shapectr = [0,0,0,0,0]
probs = [0.,0.,0.,0.,0.]
objective = [0.06,0.06,0.125,0.125,0.125]
nexamp = 5000

for i in range(nexamp):
    a = fourpix()
    if whichshape(a) > 0:
        shapectr[whichshape(a)-1] += 1
    # update probabilities based on updated counts
    for j in range(5):
        probs[j] = shapectr[j] / float(nexamp)
    if i % 100 == 0:
        #print ssd(probs,objective)
        plt.scatter(i, ssd(probs,objective), s=60, c="#348ABD")
#print shapectr
plt.show()


# In[ ]:



