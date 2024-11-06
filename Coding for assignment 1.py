import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Load dataset
file_path = 'Laptop-Price.csv'
laptop_data = pd.read_csv(file_path)

# Data Cleaning
# Drop irrelevant column
laptop_data = laptop_data.drop(columns=['Unnamed: 16'])

# Convert 'Ram' to integer by removing 'GB' suffix
laptop_data['Ram'] = laptop_data['Ram'].str.replace('GB', '').astype(int)

# Convert 'Cpu Rate' to float by removing 'GHz' suffix
laptop_data['Cpu Rate'] = laptop_data['Cpu Rate'].str.replace('GHz', '').astype(float)

# Fill missing values in storage columns and convert them to integers
laptop_data[['SSD', 'HDD', 'Flash Storage', 'Hybrid']] = laptop_data[['SSD', 'HDD', 'Flash Storage', 'Hybrid']].fillna(0).astype(int)

# Define custom color palette
custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_price_distribution(data):
    """
    Plot the distribution of laptop prices with KDE.

    Args:
        data (pd.DataFrame): DataFrame containing laptop data with 'Price_euros' column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Price_euros'], bins=30, kde=True, color="#1f77b4", edgecolor="black")
    plt.title('Distribution of Laptop Prices', fontweight='bold')
    plt.xlabel('Price in Euros', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.annotate("Peak Frequency", xy=(1000, 160), xytext=(1500, 170),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, fontweight='bold')
    plt.show()

def plot_price_by_type(data):
    """
    Plot boxplot of laptop prices by type.

    Args:
        data (pd.DataFrame): DataFrame containing laptop data with 'TypeName' and 'Price_euros' columns.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='TypeName', y='Price_euros', data=data, palette=custom_palette)
    plt.title('Price Distribution by Laptop Type', fontweight='bold')
    plt.xlabel('Laptop Type', fontweight='bold')
    plt.ylabel('Price in Euros', fontweight='bold')
    plt.xticks(rotation=45)
    plt.annotate("Highest Price Variation", xy=(3, 4000), xytext=(1, 5000),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, fontweight='bold')
    plt.show()

def plot_price_vs_screen_size(data):
    """
    Plot scatterplot of laptop prices vs screen size.

    Args:
        data (pd.DataFrame): DataFrame containing laptop data with 'Inches' and 'Price_euros' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Inches', y='Price_euros', data=data, color=custom_palette[2], alpha=0.7)
    plt.title('Price vs Screen Size (Inches)', fontweight='bold')
    plt.xlabel('Screen Size (Inches)', fontweight='bold')
    plt.ylabel('Price in Euros', fontweight='bold')
    plt.annotate("Largest Screen Size", xy=(17, 5000), xytext=(16, 6000),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, fontweight='bold')
    plt.show()

def plot_avg_price_by_ram(data):
    """
    Plot bar chart of average laptop prices by RAM size.

    Args:
        data (pd.DataFrame): DataFrame containing laptop data with 'Ram' and 'Price_euros' columns.
    """
    plt.figure(figsize=(10, 6))
    ram_price = data.groupby('Ram')['Price_euros'].mean().reset_index()
    sns.barplot(x='Ram', y='Price_euros', data=ram_price, palette=custom_palette)
    plt.title('Average Price by RAM Size', fontweight='bold')
    plt.xlabel('RAM (GB)', fontweight='bold')
    plt.ylabel('Average Price in Euros', fontweight='bold')
    plt.annotate("High Cost for High RAM", xy=(7, 4000), xytext=(5, 4500),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, fontweight='bold')
    plt.show()

def plot_avg_price_by_brand(data):
    """
    Plot bar chart of average laptop prices by brand.

    Args:
        data (pd.DataFrame): DataFrame containing laptop data with 'Company' and 'Price_euros' columns.
    """
    plt.figure(figsize=(12, 8))
    brand_price = data.groupby('Company')['Price_euros'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='Price_euros', y='Company', data=brand_price, palette=custom_palette)
    plt.title('Average Price by Laptop Brand', fontweight='bold')
    plt.xlabel('Average Price in Euros', fontweight='bold')
    plt.ylabel('Brand', fontweight='bold')
    plt.annotate("Premium Brand", xy=(3000, 0), xytext=(2500, 2),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, fontweight='bold')
    plt.show()

# Statistical Analysis
# Descriptive statistics
describe_data = laptop_data.describe()

# Correlation matrix
correlation_matrix = laptop_data.corr(numeric_only=True)

# Kurtosis and Skewness for price distribution
price_kurtosis = kurtosis(laptop_data['Price_euros'], fisher=True)
price_skewness = skew(laptop_data['Price_euros'])

# Display results
print("Descriptive Statistics:\n", describe_data)
print("\nCorrelation Matrix:\n", correlation_matrix)
print("\nPrice Kurtosis:", price_kurtosis)
print("Price Skewness:", price_skewness)

# Generate plots
plot_price_distribution(laptop_data)
plot_price_by_type(laptop_data)
plot_price_vs_screen_size(laptop_data)
plot_avg_price_by_ram(laptop_data)
plot_avg_price_by_brand(laptop_data)