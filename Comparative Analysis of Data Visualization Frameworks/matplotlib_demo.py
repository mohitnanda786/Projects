import matplotlib.pyplot as plt
import numpy as np

# 1. Generate Data
# We will create data for a simple sine wave and cosine wave
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y1 = np.sin(x)
y2 = np.cos(x)

# 2. Create the Plot
# We use the object-oriented approach (easier for complex plots)
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Add Data to Plot
ax.plot(x, y1, label='Sine Wave', color='blue', linestyle='--')
ax.plot(x, y2, label='Cosine Wave', color='red', linewidth=2)

# 4. Customize the Plot
ax.set_title('Trigonometric Functions Analysis', fontsize=16)
ax.set_xlabel('Time (t)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.legend()  # Show the legend based on 'label' arguments above
ax.grid(True) # Add a grid for readability

# 5. Save the Plot
print("Saving plot to matplotlib_plot.png...")
plt.savefig('matplotlib_plot.png')
print("Done.")

