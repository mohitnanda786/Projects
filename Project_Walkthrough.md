# Lab Assignment 1: Step-by-Step Project Walkthrough

This guide will help you execute the practical parts of your Lab Assignment. We have created two Python scripts in this folder for you to run, and provided instructions for the manual parts.

## Part 1: Python Data Visualization (Matplotlib & Seaborn)

### Step 1: Install Required Libraries
Open your terminal (Command Prompt or PowerShell) and run the following command to install the necessary tools:

```bash
pip install matplotlib seaborn pandas numpy
```

### Step 2: Run the Matplotlib Demo
I have created a file named `matplotlib_demo.py` in this folder. It generates a scientific-style plot of sine/cosine waves.

**To run it:**
1. Open your terminal in this folder (`d:\Projects\Project 1`).
2. Run:
   ```bash
   python matplotlib_demo.py
   ```
3. A window will pop up showing the graph. Analyze how the code controls the line colors, labels, and grid.

### Step 3: Run the Seaborn Demo
I have created a file named `seaborn_demo.py`. It uses a built-in dataset about penguins to show statistical relationships.

**To run it:**
1. Run:
   ```bash
   python seaborn_demo.py
   ```
2. Observe how it automatically handles colors and legends for different species.

---

## Part 2: Business Intelligence Tools (Tableau & Power BI)

Since these are graphical tools, you cannot run a script. Follow these steps to create the visual examples for your report.

### Step 4: Tableau Walkthrough
**Goal**: Create a Sales Map.

1.  **Download**: [Tableau Public](https://public.tableau.com/en-us/s/download) (Free).
2.  **Get Data**: Download the "Sample - Superstore" dataset (usually included with Tableau).
3.  **Action**:
    *   Open Tableau and connect to the Excel file.
    *   On the bottom left, click "Sheet 1".
    *   In the Data pane (left), find "State" (under Location). Double-click it.
    *   *Result*: A map of the country appears.
    *   Drag "Sales" from the Data pane onto "Color" in the Marks card.
    *   *Result*: States are colored by sales volume.
4.  **Capture**: Take a screenshot of this map for your assignment.

### Step 5: Power BI Walkthrough
**Goal**: Create a Revenue Bar Chart.

1.  **Download**: [Power BI Desktop](https://powerbi.microsoft.com/en-us/desktop/) (Free).
2.  **Get Data**: Use the same "Sample - Superstore" or any simple CSV file.
3.  **Action**:
    *   Click "Home" > "Get Data" > "Text/CSV" (or Excel).
    *   Load your data.
    *   In the "Visualizations" pane (right), click the "Clustered Bar Chart" icon.
    *   Drag "Category" to the **X-axis** field.
    *   Drag "Sales" (or Revenue) to the **Y-axis** field.
    *   *Result*: A bar chart showing sales by category.
4.  **Capture**: Take a screenshot for your assignment.

---

## Part 3: Finalizing Your Report

1.  Take screenshots of the Python windows you ran in Part 1.
2.  Take screenshots of the tools you used in Part 2.
3.  Insert them into the `Lab_Assignment_1.md` file (or your Word doc) under the "Visual Examples" sections.
