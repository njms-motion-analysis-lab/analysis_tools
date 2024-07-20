import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data structure
data = {
    'SES': ['High', 'Low', 'Medium', 'High', 'Low'],
    'Initial Diagnosis': [3, 1, 2, 3, 1],
    'Transition to Bracing': [4, 2, 3, 4, 2],
    'Experience with Bracing': [5, 3, 4, 5, 3],
    'Physical Comfort': [4, 2, 3, 4, 2],
    'Emotional Comfort': [3, 1, 2, 3, 1],
    'Support System': [5, 2, 4, 5, 2],
    'Healthcare Experience': [4, 1, 3, 4, 1],
    'Suggestions': [3, 2, 3, 3, 2]
}

df = pd.DataFrame(data)

# Example Heatmap
plt.figure(figsize=(10, 8))
heatmap_data = df.groupby('SES').mean()
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Dimensions by SES')
plt.show()

# Example Bar Chart
bar_chart_data = df.melt(id_vars=['SES'], var_name='Dimension', value_name='Score')
plt.figure(figsize=(14, 8))
sns.barplot(x='Dimension', y='Score', hue='SES', data=bar_chart_data)
plt.title('Comparison of Dimensions by SES')
plt.xticks(rotation=45)
plt.show()