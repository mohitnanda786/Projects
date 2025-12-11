import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load Data
# Seaborn has built-in datasets we can use for practice
print("Loading dataset...")
df = sns.load_dataset("penguins")

# 2. Create the Visualization
# usage: we want to see the relationship between flipper length and body mass
print("Creating JointPlot...")
g = sns.jointplot(
    data=df,
    x="flipper_length_mm",
    y="body_mass_g",
    hue="species",      # Color dots by species
    kind="scatter",     # Scatter plot
    palette="viridis",
    height=8
)

# 3. Customize
g.fig.suptitle("Penguin Analysis: Flipper Length vs Body Mass", y=1.02)

# 4. Show
print("Displaying plot... close the window to finish.")
plt.show()
