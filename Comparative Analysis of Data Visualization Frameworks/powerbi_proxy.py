import pandas as pd
import matplotlib.pyplot as plt

def generate_powerbi_proxy():
    print("Loading Sample - Superstore.xls...")
    try:
        # Load the Excel file
        df = pd.read_excel('sample_-_superstore.xls')
        
        print("Data loaded.")
        
        if 'Category' not in df.columns or 'Sales' not in df.columns:
            print("Error: Required columns 'Category' or 'Sales' not found.")
            return

        # Group by Category and Sum Sales
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        
        # Plot
        print("Creating visualization...")
        fig, ax = plt.subplots(figsize=(10, 6))
        category_sales.plot(kind='bar', color='orange', ax=ax)
        
        ax.set_title('Total Revenue by Product Category', fontsize=16)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Revenue', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Format Y axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = 'powerbi_proxy_output.png'
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_powerbi_proxy()
