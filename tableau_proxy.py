import pandas as pd
import matplotlib.pyplot as plt

def generate_tableau_proxy():
    print("Loading Sample - Superstore.xls...")
    try:
        # Load the Excel file
        # 'Sales' and 'State' are standard columns in Superstore
        df = pd.read_excel('sample_-_superstore.xls')
        
        print("Data loaded. Columns:", df.columns.tolist())
        
        # Verify columns exist
        if 'State' not in df.columns or 'Sales' not in df.columns:
            print("Error: Required columns 'State' or 'Sales' not found.")
            return

        # Group by State and Sum Sales
        state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=True)
        
        # We'll take top 15 for readability in a static chart (since we can't do a full interactive map easily)
        top_states = state_sales.tail(15)

        # Plot
        print("Creating visualization...")
        fig, ax = plt.subplots(figsize=(12, 8))
        top_states.plot(kind='barh', color='teal', ax=ax)
        
        ax.set_title('Total Sales by State (Top 15)', fontsize=16)
        ax.set_xlabel('Sales', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Format X axis as currency
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        
        output_file = 'tableau_proxy_output.png'
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_tableau_proxy()
