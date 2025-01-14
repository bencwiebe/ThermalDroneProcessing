import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from plotnine import *
import warnings
warnings.filterwarnings("ignore")
from statsmodels.formula.api import ols
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import random
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats

def analyze_treatment_impact(df, thermal_column='thermal_value_around_date_mean', pixel_column='num_pixels', masking_stringency='unspecified'):
    # Drop missing values to ensure valid statistical tests for treatment, adaptation, and thermal columns
    df = df.dropna(subset=['treatment', 'adaptation', thermal_column])

    # Initialize dictionaries to store mean values
    treatment_impacts = {'cool': {}, 'warm': {}, 'overall': {}}
    
    output_file = f'/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/{masking_stringency}_stats.txt'

    with open(output_file, 'w') as file:
        # Perform the t-test and calculate mean differences for each treatment group and adaptation type (thermal values)
        for treatment in df['treatment'].unique():
            # Subset the data for each treatment
            df_treatment = df[df['treatment'] == treatment]

            # Separate data by adaptation (cool vs warm)
            cool_group = df_treatment[df_treatment['adaptation'] == 'cool'][thermal_column]
            warm_group = df_treatment[df_treatment['adaptation'] == 'warm'][thermal_column]
            
            # Calculate means
            mean_cool = cool_group.mean()
            mean_warm = warm_group.mean()

            t_stat, p_value, _ = sm.stats.ttest_ind(cool_group, warm_group, usevar='unequal')

            # Calculate the mean difference between cool and warm (thermal)
            mean_difference = mean_cool - mean_warm

            # Store the mean values in the dictionary for treatment impact analysis
            treatment_impacts['cool'][treatment] = mean_cool
            treatment_impacts['warm'][treatment] = mean_warm

            # Write thermal t-test and mean difference results to file
            file.write(f"T-test results for {treatment} (Thermal Values):\n")
            file.write(f"Cool group mean: {mean_cool:.4f}\n")
            file.write(f"Warm group mean: {mean_warm:.4f}\n")
            file.write(f"Mean difference (Cool - Warm): {mean_difference:.4f}\n")
            file.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n")
            file.write('-' * 40 + '\n')

        # Now calculate the overall impact of treatment by averaging across treatments for each adaptation type (thermal)
        mean_cool_droughted = treatment_impacts['cool'].get('Droughted', float('nan'))
        mean_cool_irrigated = treatment_impacts['cool'].get('Irrigated', float('nan'))
        mean_warm_droughted = treatment_impacts['warm'].get('Droughted', float('nan'))
        mean_warm_irrigated = treatment_impacts['warm'].get('Irrigated', float('nan'))

        # Calculate the average treatment effect within each adaptation group (thermal)
        cool_treatment_effect = mean_cool_droughted - mean_cool_irrigated
        warm_treatment_effect = mean_warm_droughted - mean_warm_irrigated

        # Calculate overall treatment effect across both adaptation types (thermal)
        overall_mean_droughted = df[df['treatment'] == 'Droughted'][thermal_column].mean()
        overall_mean_irrigated = df[df['treatment'] == 'Irrigated'][thermal_column].mean()
        overall_treatment_effect = overall_mean_droughted - overall_mean_irrigated

        # Write overall thermal treatment impact to file
        file.write(f"Average treatment impact for cool adaptation (Thermal): {cool_treatment_effect:.4f}\n")
        file.write(f"Average treatment impact for warm adaptation (Thermal): {warm_treatment_effect:.4f}\n")
        file.write(f"Overall treatment impact (Thermal): {overall_treatment_effect:.4f}\n")
        file.write('=' * 40 + '\n')

        # ------------------- ANOVA for Pixel Data -------------------
        # Drop rows with NaNs in the columns involved in ANOVA
        df_anova = df.dropna(subset=[pixel_column, 'treatment', 'adaptation'])

        # One-way ANOVA for 'num_pixels' across combinations of 'treatment' and 'adaptation'
        model = ols(f'{pixel_column} ~ C(treatment) + C(adaptation)', data=df_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Write the ANOVA results to file
        file.write("ANOVA Results for Number of Pixels:\n")
        file.write(anova_table.to_string() + '\n')

        # If significant, perform post-hoc Tukey's HSD test for multiple comparisons
        if anova_table["PR(>F)"].iloc[0] < 0.05:
            tukey = pairwise_tukeyhsd(df_anova[pixel_column], df_anova['treatment'] + "_" + df_anova['adaptation'])
            file.write("\nTukey's HSD Post-hoc Test for Number of Pixels:\n")
            file.write(tukey.summary().as_text() + '\n')

    return treatment_impacts



# Function to remove outliers using the IQR method
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Create the KDE plot colored by adaptation and faceted by treatment
def create_kde_plot(df, thermal_column='thermal_value_around_date_mean', masking_stringency="unspecified"):
    # Combine treatment and adaptation for color/fill differentiation
    df['treatment_adaptation'] = df['treatment'] + "_" + df['adaptation']

    for treatment_adaptation in df['treatment_adaptation'].unique():
        filtered_data = df[df['treatment_adaptation'] == treatment_adaptation][thermal_column]
        filtered_data = remove_outliers_iqr(filtered_data)
        kurtosis = stats.kurtosis(filtered_data, fisher = False)
        print(f'Kurtosis for {treatment_adaptation} = {kurtosis}')
        
    for treatment_adaptation in df['treatment_adaptation'].unique():
        filtered_data = df[df['treatment_adaptation'] == treatment_adaptation][thermal_column]
        filtered_data = remove_outliers_iqr(filtered_data)
        skewness = stats.skew(filtered_data)
        print(f'Skewness for {treatment_adaptation} = {skewness}')
        
    for adaptation in df['adaptation'].unique():
        filtered_data = df[df['adaptation'] == adaptation][thermal_column]
        filtered_data = remove_outliers_iqr(filtered_data)
        skewness = stats.skew(filtered_data)
        print(f'Skewness for {adaptation} = {skewness}')

    # KDE plot
    kde_plot = (
        ggplot(df, aes(x=thermal_column, color='treatment_adaptation'))
        + geom_density(alpha=0.5)  # Alpha to control transparency for fill
        + theme_bw()  # Publication-ready theme
        + labs(
            title="Leaf temperature distribution around date mean",
            x="Thermal Value around Date Mean",
            y="Density",
            color="Treatment & Adaptation",
            fill="Treatment & Adaptation"
        )
        + xlim(-15, 25)
        + theme(
            axis_title=element_text(size=12),  # Increase axis title size
            axis_text=element_text(size=10),   # Increase axis text size
            plot_title=element_text(size=14, face='bold', ha='center'),  # Title formatting
            legend_position='right',  # Move legend to the right
            strip_text=element_text(size=12, face='bold')  # Facet title formatting
        )
    )

    # Boxplot
    boxplot = (
        ggplot(df, aes(x='adaptation', y=thermal_column, color='treatment', fill='treatment'))
        + geom_boxplot(alpha=0.5)  # Boxplot with transparency for fill
        + theme_bw()  # Publication-ready theme
        + ylim(-15,15)
        + labs(
            title="Leaf temperature distribution by treatment",
            x="Treatment",
            y="Thermal Value around Date Mean",
            color="Treatment & Adaptation",
            fill="Treatment & Adaptation"
        )
        + theme(
            axis_title=element_text(size=12),  # Increase axis title size
            axis_text=element_text(size=10),   # Increase axis text size
            plot_title=element_text(size=14, face='bold', ha='center'),  # Title formatting
            legend_position='right',  # Move legend to the right
            strip_text=element_text(size=12, face='bold')  # Facet title formatting
        )
    )

    # Display the KDE plot
    print(kde_plot)
    
    # Save the KDE plot to a file
#     kde_plot.save(f'/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/{masking_stringency}_KDE.png')

    # Display the Boxplot
    print(boxplot)

    # Save the Boxplot to a file
#     boxplot.save(f'/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/{masking_stringency}_Boxplot.png')

    return kde_plot, boxplot

def remove_outliers(df, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile Range

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered

# Now modify the boxplot function to include outlier removal
def create_simple_boxplot(df, num_pixels_column='num_pixels', masking_stringency = "unspecified"):
    # Remove outliers from the 'num_pixels' column
    df_filtered = remove_outliers(df, num_pixels_column)

    plot = (
        ggplot(df_filtered, aes(x='adaptation', y=num_pixels_column, color='treatment'))
        + geom_boxplot(outlier_shape=None)
        + theme_bw()  # Publication-ready theme
        + labs(
            title="Boxplot of Number of Pixels by Adaptation",
            x="Adaptation",
            y="Number of Pixels",
            color="Adaptation"
        )
    )

    # Display the plot
    print(plot)
    plot.save(f'/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/{masking_stringency}_leafPixelsBoxplot.png')

def compute_stats(df, scenario_name):
    print(scenario_name)
    # Calculate means for each group
    irrigated_warm_mean = df[(df['treatment'] == 'Irrigated') & (df['adaptation'] == 'warm')]['thermal_value_around_date_mean'].mean()
    irrigated_cool_mean = df[(df['treatment'] == 'Irrigated') & (df['adaptation'] == 'cool')]['thermal_value_around_date_mean'].mean()
    
    droughted_warm_mean = df[(df['treatment'] == 'Droughted') & (df['adaptation'] == 'warm')]['thermal_value_around_date_mean'].mean()
    droughted_cool_mean = df[(df['treatment'] == 'Droughted') & (df['adaptation'] == 'cool')]['thermal_value_around_date_mean'].mean()

    # Compute overall means
    warm_data = df[(df['adaptation'] == 'warm')]['thermal_value_around_date_mean']
    overall_warm_mean = warm_data.mean()
    print(f'overall overall_warm_mean: {overall_warm_mean}')
    cool_data = df[(df['adaptation'] == 'cool')]['thermal_value_around_date_mean']
    overall_cool_mean = cool_data.mean()
    print(f'overall overall_cool_mean: {overall_cool_mean}')
    overall_diff = overall_cool_mean - overall_warm_mean
    p_val_adaptation = stats.ttest_ind(warm_data, cool_data).pvalue
    print(f'overall diff between warm and cool: {overall_diff}, p-value: {p_val_adaptation}')
    
    overall_irrigated_data = df[(df['treatment'] == 'Irrigated')]['thermal_value_around_date_mean']
    overall_droughted_data = df[(df['treatment'] == 'Droughted')]['thermal_value_around_date_mean']
    treatment_diff = overall_droughted_data.mean() - overall_irrigated_data.mean()
    p_val_treatment = stats.ttest_ind(overall_irrigated_data, overall_droughted_data).pvalue
    print(f'overall diff between treatments: {treatment_diff}, p-value: {p_val_treatment}')
    print('\n\n')
    
    # Compute differences between warm and cool means for each treatment
    diff_irrigated = irrigated_warm_mean - irrigated_cool_mean
    diff_droughted = droughted_warm_mean - droughted_cool_mean

    # Compute treatment differences within each adaptation
    diff_treatment_warm = droughted_warm_mean - irrigated_warm_mean
    diff_treatment_cool = droughted_cool_mean - irrigated_cool_mean

    # Perform t-tests to get p-values for warm and cool comparisons between irrigated and droughted
    irrigated_warm_data = df[(df['treatment'] == 'Irrigated') & (df['adaptation'] == 'warm')]['thermal_value_around_date_mean']
    irrigated_cool_data = df[(df['treatment'] == 'Irrigated') & (df['adaptation'] == 'cool')]['thermal_value_around_date_mean']
    p_val_irrigated = stats.ttest_ind(irrigated_warm_data, irrigated_cool_data).pvalue

    droughted_warm_data = df[(df['treatment'] == 'Droughted') & (df['adaptation'] == 'warm')]['thermal_value_around_date_mean']
    droughted_cool_data = df[(df['treatment'] == 'Droughted') & (df['adaptation'] == 'cool')]['thermal_value_around_date_mean']
    p_val_droughted = stats.ttest_ind(droughted_warm_data, droughted_cool_data).pvalue

    # Perform t-tests for treatment differences within warm and cool adaptations
    p_val_treatment_warm = stats.ttest_ind(irrigated_warm_data, droughted_warm_data).pvalue
    p_val_treatment_cool = stats.ttest_ind(irrigated_cool_data, droughted_cool_data).pvalue

    return [
        scenario_name, 
        irrigated_warm_mean, irrigated_cool_mean, 
        droughted_warm_mean, droughted_cool_mean,
        diff_irrigated, diff_droughted, 
        diff_treatment_warm, diff_treatment_cool,  # Added treatment differences within each adaptation
        p_val_irrigated, p_val_droughted, 
        p_val_treatment_warm, p_val_treatment_cool  # Added p-values for treatment differences
    ]