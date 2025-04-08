#import Lib
from IPython.display import Image
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np

def create_Feature_Value(output_file):
    """
        Create feature value of quantitative value of File CSV
    :param output_file:
    :return:
    """
    # Read the CSV file
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Filter numeric columns
    quantitative_vars = df.select_dtypes(include=['number'])
    categorical_vars = df.select_dtypes(include=['object', 'category'])
    # Filter some columns that not quantitative
    quantitative_vars  = quantitative_vars.drop(columns=['Shader', 'Open_GL'])
    # Create describe table
    des_quantitative_vars = quantitative_vars.describe(include='all').map(lambda x: f"{x:.2f}")
    # Adjust some row in table
    des_quantitative_vars = des_quantitative_vars.drop('count')
    mode_row = quantitative_vars.mode().iloc[0]
    des_quantitative_vars.loc['mode'] = mode_row
    #Export to image
    dfi.export(des_quantitative_vars, output_file)
    Image(output_file)

def create_Performance_Line_Chart(output_file):
    """
        create line chart of Texture Rate, Pixel Rate, Core Speed through Release Year
    :param output_file:
    :return:
    """
    #read csv
    df=pd.read_csv('data/Cleaned_GPUs_KNN.csv')

    # Ensure 'Release_Year' is numeric and convert to integer
    df['Release_Year'] = pd.to_numeric(df['Release_Year'], errors='coerce').dropna().astype(int)

    # Drop rows with missing values in the relevant columns
    df = df.dropna(subset=['Core_Speed', 'Pixel_Rate', 'Texture_Rate', 'Release_Year'])

    # Group by Release_Year and compute mean values for Pixel_Rate and Texture_Rate
    grouped = df.groupby('Release_Year')[['Core_Speed', 'Pixel_Rate', 'Texture_Rate']].mean().reset_index()

    # Create a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['Release_Year'], grouped['Core_Speed'], marker='o', label='Core Speed (MHz)')
    plt.plot(grouped['Release_Year'], grouped['Pixel_Rate'], marker='o', label='Pixel Rate (GPixel/s)')
    plt.plot(grouped['Release_Year'], grouped['Texture_Rate'], marker='o', label='Texture Rate (GTexel/s)')

    plt.title("Performance Over Time: Core Speed, Pixel Rate and Texture Rate vs. Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("")
    plt.grid(True)
    plt.legend()
    plt.xticks(grouped['Release_Year'], rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


class Drop:
    pass


def create_3D_Scatter_Plot(output_file):
    """
        create 3d scatter plot of Memory Speed, Memory Bandwidth, Texture Rate
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    #Drop rows with missing memory data
    df = df.dropna(subset=['Memory_Speed', 'Memory_Bandwidth', 'Texture_Rate'])

    # Prepare 3D plot

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D scatter
    scatter = ax.scatter(
        df['Memory_Speed'],
        df['Memory_Bandwidth'],
        df['Texture_Rate'],
        c=df['Memory_Bandwidth'], cmap='viridis', s=40, alpha=0.8
    )

    # Axis labels
    ax.set_xlabel('Memory Speed (MHz)')
    ax.set_ylabel('Memory Bandwidth (GB/s)')
    ax.set_zlabel('Texture Rate (GTexel/s)')
    ax.set_title('3D Scatter Plot: Memory Growth')

    # Add color bar
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label("Memory Bandwidth (GB/s)")
    # Save and show the plot
    plt.savefig(output_file, dpi=300)
    plt.show()

def create_Bar_Chart_Memory_Bandwidth(output_file):
    """
        create Bar Chart of Memory Bandwidth through Year
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv("data/Cleaned_GPUs_KNN.csv")
    # Convert Release_Year to integer
    df['Release_Year'] = pd.to_numeric(df['Release_Year'], errors='coerce').dropna().astype(int)

    # Drop rows with missing values in relevant columns
    df = df.dropna(subset=['Release_Year', 'Memory_Bandwidth'])

    # Group by Release_Year and Memory_Type, compute average Memory_Speed
    grouped = df.groupby(['Release_Year'])['Memory_Bandwidth'].mean().reset_index()

    # Create bar plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='Release_Year', y='Memory_Bandwidth')

    plt.title('Memory Bandwidth Growth by Year ')
    plt.xlabel('Release Year')
    plt.ylabel('Average Memory Bandwidth (GB/s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def create_Scatter_Plot_Best_Resolution(output_file):
    """
        create Scatter Plot Best Resolution through Year
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Best_Resolution', 'Release_Year'])
    df['Release_Year'] = df['Release_Year'].astype(int)

    # Function to calculate total pixels from resolution string
    def calculate_pixels(resolution):
        width, height = resolution.split('x')
        return int(width) * int(height)

    # Add a Total_Pixels column
    df['Total_Pixels'] = df['Best_Resolution'].apply(calculate_pixels)

    # Get unique resolutions and sort by total pixels
    unique_resolutions = sorted(df['Best_Resolution'].unique(), key=calculate_pixels)

    # Create a colormap with enough unique colors
    n_resolutions = len(unique_resolutions)
    colors = plt.get_cmap('tab20')(range(min(n_resolutions, 20)))  # Use tab20, cap at 20 colors
    if n_resolutions > 20:
        colors = plt.get_cmap('viridis')(range(n_resolutions) / (n_resolutions - 1))  # Scale for more

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for i, resolution in enumerate(unique_resolutions):
        subset = df[df['Best_Resolution'] == resolution]
        plt.scatter(subset['Release_Year'], subset['Total_Pixels'],
                    color=colors[i], label=f"{resolution} ({calculate_pixels(resolution):,d} pixels)",
                    alpha=0.6, s=50)

    # Customize the plot
    plt.xlabel('Release Year')
    plt.ylabel('Best_Resolution (Total Pixels)')
    plt.title('Best_Resolution vs. Release Year: Progression of GPU Resolutions')
    plt.legend(title='Best Resolution', bbox_to_anchor=(0, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set x-axis ticks to every unique year in the dataset
    unique_years = sorted(df['Release_Year'].unique())
    plt.xticks(unique_years, rotation=45)  # Rotate labels for readability

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
def create_Scatter_Plot_Resolution_WxH(output_file):
    """
        create Scatter Plot Resolution WxH through Year
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Resolution_WxH', 'Release_Year'])
    df['Release_Year'] = df['Release_Year'].astype(int)

    # Function to calculate total pixels from resolution string
    def calculate_pixels(resolution):
        width, height = resolution.split('x')
        return int(width) * int(height)

    # Add a Total_Pixels column
    df['Total_Pixels'] = df['Resolution_WxH'].apply(calculate_pixels)

    # Get unique resolutions and sort by total pixels
    unique_resolutions = sorted(df['Resolution_WxH'].unique(), key=calculate_pixels)

    # Create a colormap with enough unique colors
    n_resolutions = len(unique_resolutions)
    colors = plt.get_cmap('tab20')(range(min(n_resolutions, 20)))  # Use tab20, cap at 20 colors
    if n_resolutions > 20:
        colors = plt.get_cmap('viridis')(range(n_resolutions) / (n_resolutions - 1))  # Scale for more

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for i, resolution in enumerate(unique_resolutions):
        subset = df[df['Resolution_WxH'] == resolution]
        plt.scatter(subset['Release_Year'], subset['Total_Pixels'],
                    color=colors[i], label=f"{resolution} ({calculate_pixels(resolution):,d} pixels)",
                    alpha=0.6, s=50)

    # Customize the plot
    plt.xlabel('Release Year')
    plt.ylabel('Resolution_WxH (Total Pixels)')
    plt.title('Resolution_WxH vs. Release Year: Progression of GPU Resolutions')
    plt.legend(title='Resolution WxH', bbox_to_anchor=(0, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set x-axis ticks to every unique year in the dataset
    unique_years = sorted(df['Release_Year'].unique())
    plt.xticks(unique_years, rotation=45)  # Rotate labels for readability

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
def create_Pie_Chart_Manufacturer(output_file):
    """
        create Pie Chart with Manufacturer
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data: Drop rows with missing Manufacturer
    df = df.dropna(subset=['Manufacturer'])

    # Calculate the count of GPUs per Manufacturer
    manufacturer_counts = df['Manufacturer'].value_counts()

    # Create the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(manufacturer_counts, labels=manufacturer_counts.index, autopct='%1.1f%%', startangle=90)

    # Customize the plot
    plt.title('Manufacturer Count of GPU type', fontsize=14, pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

    # Save and display the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
def create_Stack_Bar_Chart_Manufacturer(output_file):
    """
        create Stack Bar Chart of Manufacturer through Year
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Manufacturer', 'Release_Year'])
    df['Release_Year'] = df['Release_Year'].astype(int)

    # Prepare the data: Count GPUs by Release_Year and Manufacturer
    count_df = df.groupby(['Release_Year', 'Manufacturer']).size().unstack(fill_value=0)

    # Sort manufacturers by total count across all years (descending order)
    count_df = count_df[count_df.sum().sort_values(ascending=False).index]

    # Create a colormap with enough unique colors
    n_manufacturers = len(count_df.columns)
    colors = plt.get_cmap('tab20')(range(min(n_manufacturers, 20)))  # Use tab20, cap at 20 colors
    if n_manufacturers > 20:
        colors = plt.get_cmap('viridis')(range(n_manufacturers) / (n_manufacturers - 1))  # Scale for more

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    count_df.plot(kind='bar', stacked=True, ax=ax, color=colors)

    # Customize the plot
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Count of GPU')
    ax.set_title('Manufacturer Count of GPU Through Year')
    plt.xticks(rotation=45)
    ax.legend(title='Manufacturer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
def create_Scatter_Plot_Manufacturer(output_file):
    """
        create Scatter Plot of texture rate with Manufacturer through Year
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean and filter the data
    df = df.dropna(subset=['Manufacturer', 'Release_Year', 'Texture_Rate'])
    df['Release_Year'] = df['Release_Year'].astype(int)

    # Filter for Nvidia and AMD only
    df = df[df['Manufacturer'].isin(['Nvidia', 'AMD'])]

    # Ensure Texture_Rate is numeric (convert if necessary)
    df['Texture_Rate'] = pd.to_numeric(df['Texture_Rate'], errors='coerce')
    df = df.dropna(subset=['Texture_Rate'])  # Drop any rows where conversion failed

    # Create the scatter plot
    plt.figure(figsize=(12, 8))

    # Plot Nvidia
    nvidia = df[df['Manufacturer'] == 'Nvidia']
    plt.scatter(nvidia['Release_Year'], nvidia['Texture_Rate'],
                color='green', label='Nvidia', alpha=0.6, s=80)

    # Plot AMD
    amd = df[df['Manufacturer'] == 'AMD']
    plt.scatter(amd['Release_Year'], amd['Texture_Rate'],
                color='red', label='AMD', alpha=0.6, s=80)

    # Customize the plot
    plt.xlabel('Release Year')
    plt.ylabel('Texture Rate(GTexels/s)')
    plt.title('Performance: Nvidia vs AMD Texture Rate')
    plt.legend(title='Manufacturer')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set x-axis ticks to every unique year
    unique_years = sorted(df['Release_Year'].unique())
    plt.xticks(unique_years, rotation=45)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
def create_Normal_QQPlot_Memory_Bandwidth(output_file):
    """
        create Memory Bandwidth
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Memory_Bandwidth'])
    df['Memory_Bandwidth'] = pd.to_numeric(df['Memory_Bandwidth'], errors='coerce')
    df = df.dropna(subset=['Memory_Bandwidth'])  # Drop any rows where conversion failed

    # Create a figure with subplots: 2 rows (original and log), 2 columns (histogram and Q-Q plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Original Memory_Bandwidth ---
    # Histogram with KDE
    sns.histplot(df['Memory_Bandwidth'], kde=True, stat='density', ax=axes[0, 0])
    # Overlay a normal distribution curve
    mean = df['Memory_Bandwidth'].mean()
    std = df['Memory_Bandwidth'].std()
    x = np.linspace(df['Memory_Bandwidth'].min(), df['Memory_Bandwidth'].max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', label='Normal Fit')
    axes[0, 0].set_title('Phân Phối Memory Bandwidth (Original)')
    axes[0, 0].set_xlabel('Memory Bandwidth (GB/s)')
    axes[0, 0].set_ylabel('Mật Độ (Density)')
    axes[0, 0].legend()

    # Q-Q plot for original data
    stats.probplot(df['Memory_Bandwidth'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: Memory Bandwidth (Original)')

    # --- Log-Transformed Memory_Bandwidth ---
    # Apply log transformation (add a small constant to avoid log(0))
    df['Log_Memory_Bandwidth'] = np.log(df['Memory_Bandwidth'])

    # Histogram with KDE for log-transformed data
    sns.histplot(df['Log_Memory_Bandwidth'], kde=True, stat='density', ax=axes[1, 0])
    # Overlay a normal distribution curve
    mean_log = df['Log_Memory_Bandwidth'].mean()
    std_log = df['Log_Memory_Bandwidth'].std()
    x_log = np.linspace(df['Log_Memory_Bandwidth'].min(), df['Log_Memory_Bandwidth'].max(), 100)
    axes[1, 0].plot(x_log, stats.norm.pdf(x_log, mean_log, std_log), 'r-', label='Normal Fit')
    axes[1, 0].set_title('Phân Phối Memory Bandwidth (Log-Transformed)')
    axes[1, 0].set_xlabel('Log(Memory Bandwidth)')
    axes[1, 0].set_ylabel('Mật Độ (Density)')
    axes[1, 0].legend()

    # Q-Q plot for log-transformed data
    stats.probplot(df['Log_Memory_Bandwidth'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Memory Bandwidth (Log-Transformed)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def create_Normal_QQPlot_Texture_Rate(output_file):
    """
        create Texture Rate
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Texture_Rate'])
    df['Texture_Rate'] = pd.to_numeric(df['Texture_Rate'], errors='coerce')
    df = df.dropna(subset=['Texture_Rate'])  # Drop any rows where conversion failed

    # Create a figure with subplots: 2 rows (original and log), 2 columns (histogram and Q-Q plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Original Texture_Rate ---
    # Histogram with KDE
    sns.histplot(df['Texture_Rate'], kde=True, stat='density', ax=axes[0, 0])
    # Overlay a normal distribution curve
    mean = df['Texture_Rate'].mean()
    std = df['Texture_Rate'].std()
    x = np.linspace(df['Texture_Rate'].min(), df['Texture_Rate'].max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', label='Normal Fit')
    axes[0, 0].set_title('Phân Phối Texture Rate (Original)')
    axes[0, 0].set_xlabel('Texture Rate (GTexels/s)')
    axes[0, 0].set_ylabel('Mật Độ (Density)')
    axes[0, 0].legend()

    # Q-Q plot for original data
    stats.probplot(df['Texture_Rate'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: Texture Rate (Original)')

    # --- Log-Transformed Texture_Rate ---
    # Apply log transformation: log(Texture_Rate + 1)
    df['Log_Texture_Rate'] = np.log(df['Texture_Rate'] + 1)

    # Histogram with KDE for log-transformed data
    sns.histplot(df['Log_Texture_Rate'], kde=True, stat='density', ax=axes[1, 0])
    # Overlay a normal distribution curve
    mean_log = df['Log_Texture_Rate'].mean()
    std_log = df['Log_Texture_Rate'].std()
    x_log = np.linspace(df['Log_Texture_Rate'].min(), df['Log_Texture_Rate'].max(), 100)
    axes[1, 0].plot(x_log, stats.norm.pdf(x_log, mean_log, std_log), 'r-', label='Normal Fit')
    axes[1, 0].set_title('Phân Phối Texture Rate (Log-Transformed)')
    axes[1, 0].set_xlabel('Log(Texture Rate + 1)')
    axes[1, 0].set_ylabel('Mật Độ (Density)')
    axes[1, 0].legend()

    # Q-Q plot for log-transformed data
    stats.probplot(df['Log_Texture_Rate'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Texture Rate (Log-Transformed)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
def create_Normal_QQPlot_L2_Cache(output_file):
    """
        create L2 cache
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['L2_Cache'])
    df['L2_Cache'] = pd.to_numeric(df['L2_Cache'], errors='coerce')
    df = df.dropna(subset=['L2_Cache'])  # Drop any rows where conversion failed

    # Create a figure with subplots: 2 rows (original and log), 2 columns (histogram and Q-Q plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Original L2_Cache ---
    # Histogram with KDE
    sns.histplot(df['L2_Cache'], kde=True, stat='density', ax=axes[0, 0])
    # Overlay a normal distribution curve
    mean = df['L2_Cache'].mean()
    std = df['L2_Cache'].std()
    x = np.linspace(df['L2_Cache'].min(), df['L2_Cache'].max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', label='Normal Fit')
    axes[0, 0].set_title('Phân Phối L2 Cache (Original)')
    axes[0, 0].set_xlabel('L2 Cache (KB)')  # Adjust unit if needed (e.g., MB)
    axes[0, 0].set_ylabel('Mật Độ (Density)')
    axes[0, 0].legend()

    # Q-Q plot for original data
    stats.probplot(df['L2_Cache'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: L2 Cache (Original)')

    # --- Log-Transformed L2_Cache ---
    # Apply log transformation: log(L2_Cache + 1)
    df['Log_L2_Cache'] = np.log(df['L2_Cache'] + 1)

    # Histogram with KDE for log-transformed data
    sns.histplot(df['Log_L2_Cache'], kde=True, stat='density', ax=axes[1, 0])
    # Overlay a normal distribution curve
    mean_log = df['Log_L2_Cache'].mean()
    std_log = df['Log_L2_Cache'].std()
    x_log = np.linspace(df['Log_L2_Cache'].min(), df['Log_L2_Cache'].max(), 100)
    axes[1, 0].plot(x_log, stats.norm.pdf(x_log, mean_log, std_log), 'r-', label='Normal Fit')
    axes[1, 0].set_title('Phân Phối L2 Cache (Log-Transformed)')
    axes[1, 0].set_xlabel('Log(L2 Cache + 1)')
    axes[1, 0].set_ylabel('Mật Độ (Density)')
    axes[1, 0].legend()

    # Q-Q plot for log-transformed data
    stats.probplot(df['Log_L2_Cache'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: L2 Cache (Log-Transformed)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
def create_Normal_QQPlot_Max_Power(output_file):
    """
        create max power
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    # Clean the data
    df = df.dropna(subset=['Max_Power'])
    df['Max_Power'] = pd.to_numeric(df['Max_Power'], errors='coerce')
    df = df.dropna(subset=['Max_Power'])  # Drop any rows where conversion failed

    # Create a figure with subplots: 2 rows (original and log), 2 columns (histogram and Q-Q plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Original Max_Power ---
    # Histogram with KDE
    sns.histplot(df['Max_Power'], kde=True, stat='density', ax=axes[0, 0])
    # Overlay a normal distribution curve
    mean = df['Max_Power'].mean()
    std = df['Max_Power'].std()
    x = np.linspace(df['Max_Power'].min(), df['Max_Power'].max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', label='Normal Fit')
    axes[0, 0].set_title('Phân Phối Max Power (Original)')
    axes[0, 0].set_xlabel('Max Power (W)')
    axes[0, 0].set_ylabel('Mật Độ (Density)')
    axes[0, 0].legend()

    # Q-Q plot for original data
    stats.probplot(df['Max_Power'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: Max Power (Original)')

    # --- Log-Transformed Max_Power ---
    # Apply log transformation: log(Max_Power + 1)
    df['Log_Max_Power'] = np.log(df['Max_Power'] + 1)

    # Histogram with KDE for log-transformed data
    sns.histplot(df['Log_Max_Power'], kde=True, stat='density', ax=axes[1, 0])
    # Overlay a normal distribution curve
    mean_log = df['Log_Max_Power'].mean()
    std_log = df['Log_Max_Power'].std()
    x_log = np.linspace(df['Log_Max_Power'].min(), df['Log_Max_Power'].max(), 100)
    axes[1, 0].plot(x_log, stats.norm.pdf(x_log, mean_log, std_log), 'r-', label='Normal Fit')
    axes[1, 0].set_title('Phân Phối Max Power (Log-Transformed)')
    axes[1, 0].set_xlabel('Log(Max Power + 1)')
    axes[1, 0].set_ylabel('Mật Độ (Density)')
    axes[1, 0].legend()

    # Q-Q plot for log-transformed data
    stats.probplot(df['Log_Max_Power'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Max Power (Log-Transformed)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
def create_Matrix_Correlation(output_file, m_method: "" = "pearson"):
    """
        create Matrix Correlation with pearson
    :param m_method: pearson, kendall, spearman
    :param output_file:
    :return:
    """
    #read csv
    df = pd.read_csv('data/Cleaned_GPUs_KNN.csv')
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr(m_method)

    # Compute the average absolute correlation for each column
    avg_corr = corr_matrix.sum().sort_values(ascending=False)

    # Reorder the columns (and rows) based on the sorted average correlation
    sorted_corr_matrix = corr_matrix.loc[avg_corr.index, avg_corr.index]

    # Display the sorted correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Sorted Correlation Matrix")
    plt.savefig(output_file, dpi=300)
    plt.show()
# main start from here
create_Feature_Value("Chart/Thong_Ke_Mo_Ta/Tong quan du lieu.png")
create_3D_Scatter_Plot("Chart/Thong_Ke_Mo_Ta/3D Scatter.png")
create_Pie_Chart_Manufacturer("Chart/Thong_Ke_Mo_Ta/GPU by Manufacturer")
create_Performance_Line_Chart("Chart/Thong_Ke_Mo_Ta/Performance Line Chart")
create_Bar_Chart_Memory_Bandwidth("Chart/Thong_Ke_Mo_Ta/Memoy Bandwidth")
create_Normal_QQPlot_Texture_Rate("Chart/Thong_Ke_Mo_Ta/Texture Rate QQ-Plot")
create_Normal_QQPlot_L2_Cache("Chart/Thong_Ke_Mo_Ta/L2_Cache QQ-Plot")
create_Normal_QQPlot_Memory_Bandwidth("Chart/Thong_Ke_Mo_Ta/Memory Bandwidth QQ-Plot")
create_Normal_QQPlot_Max_Power("Chart/Thong_Ke_Mo_Ta/Max Power QQ-Plot")
create_Scatter_Plot_Best_Resolution("Scatter Plot for Resolution")
create_Matrix_Correlation("Chart/Thong_Ke_Mo_Ta/Correlation Matrix")
create_Scatter_Plot_Manufacturer("Chart/Thong_Ke_Mo_Ta/Scatter plot by Manufacturer")
create_Scatter_Plot_Resolution_WxH("Chart/Thong_Ke_Mo_Ta/Scatter plot for Resolution")
create_Scatter_Plot_Best_Resolution("Chart/Thong_Ke_Mo_Ta/Scatter plot for Best Resolution")