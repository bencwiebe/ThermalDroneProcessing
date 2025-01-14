import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from plotnine import *
import warnings
from statsmodels.formula.api import ols
import statsmodels.api as sm
import random
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats

# customized functions for this analysis from other python file
from functions import *

# Set up directory and list of files
#this one was reprocessed using thermal studio pro with the correct air temperatures
root_dir = '/Users/bcw269/Documents/Common_gardens/thermal2024/thermalFlattened_processed_40c' 

files = os.listdir(root_dir)
filtered_files = [file for file in files if " (" not in file]
jpg_files = [file for file in filtered_files if file.endswith('jpg')]

# Initialize dataframe to store results
df = pd.DataFrame(columns=['population', 'block', 'date', 'thermal_value', 'file_number','num_pixels'])
dates = []

#loop through different stringencies for sensitivity analysis
stringencies = ['baseline', 'conservative', 'extra_conservative'] 

for stringency in stringencies:
    
    print(stringency)
    ctr = 0
    
    for file in jpg_files:
        
        ctr = ctr + 1
        if ctr %100 == 0:
            print(ctr)
            
        try:
            # Extract metadata from filename
            date = file.split('_')[1]
            if date not in dates:
                dates.append(date)
            year = file.split('_')[2]
            block = file.split('_')[3]
            block = 'block2' if block == 'b2' else 'block3' if block == 'b3' else block
            population = file.split('_')[4]
            file_number = file.split('_')[5]

            # Paths to RGB and thermal CSV
            rgb_path = os.path.join(root_dir, file)
            csv_path = rgb_path.replace('jpg', 'csv')

            # Load RGB image and split channels
            rgb = np.array(Image.open(rgb_path))
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

            if stringency == 'baseline':
                #baseline
                mask = g > r  # Green is greater than red
                mask2 = g > 100  # Green is bright enough, eliminating shadows
                mask3 = (g < 120) | (r < 120) | (b < 120) # elimitate bright spots by making sure at least one of the colors is below 120
                mask4 = (g < 120) | (r < 120) | (b < 120) # make mask four the same as three just avoid issues down the line
                #                 mask4 = (g * 2 - r - b) > 30  # Green dominance in relation to red and blue
            elif stringency == 'conservative':
            #conservative
                mask = g > (r+10)  # Green is greater than red
                mask2 = g > 100  # Green is bright enough, eliminating shadows
                mask3 = (g < 120) | (r < 120) | (b < 120) # elimitate bright spots by making sure at least one of the colors is below 120
                mask4 = (g * 2 - r - b) > 40  # Green dominance in relation to red and blue
            elif stringency == 'extra_conservative':
            #conservative
                mask = g > (r+20)  # Green is greater than red
                mask2 = g > 100  # Green is bright enough, eliminating shadows
                mask3 = (g < 120) | (r < 120) | (b < 120) # elimitate bright spots by making sure at least one of the colors is below 120
                mask4 = (g * 2 - r - b) > 50  # Green dominance in relation to red and blue
            elif stringency == 'liberal':
    #         liberal
                mask = g > (r-10)  # Green is greater than red
                mask2 = g > 100  # Green is bright enough, eliminating shadows
                mask3 = (g < 120) | (r < 120) | (b < 120) # elimitate bright spots by making sure at least one of the colors is below 120
                mask4 = (g * 2 - r - b) > 20  # Green dominance in relation to red and blue
            else:
                print('you mispelled something')
                break
                
            # Combine masks to keep areas that match the conditions
            final_mask = mask & mask2 & mask3 & mask4

            # Apply mask to the RGB image
            masked_rgb = np.where(final_mask[..., None], rgb, [0, 0, 0])

            # Load thermal data
            thermal_data = pd.read_csv(csv_path, skiprows=10, header=None).drop(0, axis=1).values

            # Apply the mask to the thermal image and keep only valid pixels
            masked_thermal = np.where(final_mask, thermal_data, np.nan)

            # Sample up to 100 remaining pixels from the masked thermal image
            thermal_pixels = masked_thermal[~np.isnan(masked_thermal)].flatten()
            num_pixels = len(thermal_pixels)
            thermal_mean = thermal_pixels.mean()

            sampled_pixels = random.sample(list(thermal_pixels), min(100, len(thermal_pixels)))

            # Add the sampled data to the dataframe
            temp_df = pd.DataFrame({
                'population': population,
                'block': block,
                'date': date,
                'thermal_value': sampled_pixels,
                'file_number': file_number,
                'num_pixels': num_pixels,
                'thermal_file_mean': thermal_mean
            })
            df = pd.concat([df, temp_df], ignore_index=True)

            ''' ## Uncomment this if you want to visually inspect the masking
            # Display images for visual inspection
            vmin = np.nanpercentile(thermal_data, 5)  # Use np.nanpercentile to handle NaNs
            vmax = np.nanpercentile(thermal_data, 95)

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            # Display the images without tick marks
            axs[0, 0].imshow(rgb)
            axs[0, 0].set_title('RGB')
            axs[0, 0].set_xticks([])  # Remove x-axis ticks
            axs[0, 0].set_yticks([])  # Remove y-axis ticks

            axs[0, 1].imshow(masked_rgb)
            axs[0, 1].set_title('Masked RGB')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[1, 0].imshow(thermal_data, cmap='jet', vmin=vmin, vmax=vmax)
            axs[1, 0].set_title('Thermal')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])

            axs[1, 1].imshow(masked_thermal, cmap='jet', vmin=vmin, vmax=vmax)
            axs[1, 1].set_title('Masked Thermal')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            # Adjust the layout to make everything closer together
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce the space between subplots

            plt.tight_layout()
            plt.show()
            '''
        except Exception as e:
            # print(f"Error processing {file}: {e}")
            continue
    
    #combine everything
    #a few have weird naming conventions -- selecting only those that end with jpg
    df = df[df['file_number'].str.endswith('.jpg')]

    # Calculate the mean 'thermal_file_mean' by 'date' and add to the original dataframe
    df['thermal_date_mean'] = df.groupby('date')['thermal_file_mean'].transform('mean')
    df['thermal_value_around_date_mean'] = df['thermal_value'] - df['thermal_date_mean'] #normalize thermal values around average date mean

    # Calculate the mean 'thermal_value' by 'date' using all sampled pixels and add to the original dataframe
    df['thermal_date_mean'] = df.groupby('date')['thermal_value'].transform('mean')
    df['thermal_value_aroundDatePixelMean'] = df['thermal_value'] - df['thermal_date_mean']  # Normalize thermal values around date pixel mean

    # convert things to int
    df['num_pixels'] = df['num_pixels'].astype(int)
    df['thermal_file_mean'] = df['thermal_file_mean'].astype(float)

    # Define your block mappings to treatments
    treatment_mapping = {
        'block2': 'Droughted',
        'block3': 'Irrigated'
    }

    #define adapation... from Cooper et al. 2019
    adaptation_mapping = {
        "KKH": "cool",
        "NRV": "warm",
        "CCR": "warm",
        "CLF": "cool",
        "JLA": "cool",
        "LBW": "warm",
        'LBW2': 'warm'
    }

    # Map the block to treatment types
    df['treatment'] = df['block'].map(treatment_mapping)
    df['adaptation'] = df['population'].map(adaptation_mapping)
    df = df.dropna(subset=['adaptation'])

    # Group by 'file_number' and aggregate
    df_grouped = df.groupby('file_number').agg({
        'thermal_value': 'mean',
        'population': 'first',
        'block': 'first',
        'date': 'first',
        'num_pixels': 'first',
        'thermal_file_mean': 'first',
        'thermal_value_around_date_mean': 'mean',
        'thermal_value_aroundDatePixelMean':'mean',
        'treatment': 'first',
        'adaptation': 'first'
    }).reset_index()

    # Ensure correct types in the grouped dataframe
    df_grouped['num_pixels'] = df_grouped['num_pixels'].astype(int)
    df_grouped['thermal_file_mean'] = df_grouped['thermal_file_mean'].astype(float)

    # Normalize 'thermal_mean' in the grouped dataframe using the mean by 'date'
    df_grouped = df_grouped.merge(
        df[['date', 'thermal_date_mean']].drop_duplicates(), 
        on='date', 
        how='left'
    )
    df_grouped['thermal_value_around_date_mean'] = df_grouped['thermal_value'] - df_grouped['thermal_date_mean']
    
    if stringency == 'baseline':
        df_baseline = df
        df_grouped_baseline = df_grouped
    elif stringency == 'conservative':
        df_conservative = df
        df_grouped_conservative = df_grouped
    elif stringency == 'extra_conservative':
        df_extra_conservative = df
        df_grouped_extra_conservative = df_grouped
    elif stringency == 'liberal':
        df_liberal = df
        df_grouped_liberal = df_grouped    
    else:
        print('you misspelled something')
        break


#do stats and make graphs for baseline
df_baseline.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_baselineMasking.csv')
df_grouped_baseline.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_baselineMasking_imageWise.csv')
analyze_treatment_impact(df_grouped_baseline, masking_stringency = 'baseline')
create_kde_plot(df_grouped_baseline, thermal_column='thermal_value_aroundDatePixelMean', masking_stringency = 'baseline')
create_simple_boxplot(df_grouped_baseline, masking_stringency = 'baseline')

#do stats and make graphs for conservative
df_conservative.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_conservativeMasking.csv')
df_grouped_conservative.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_conservativeMasking_imageWise.csv')
analyze_treatment_impact(df_grouped_conservative, masking_stringency = 'conservative')
create_kde_plot(df_conservative, thermal_column='thermal_value_aroundDatePixelMean', masking_stringency = 'conservative')
create_simple_boxplot(df_conservative, masking_stringency = 'conservative')

#do stats and make graphs for extra_conservative
df_extra_conservative.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_extraConservativeMasking.csv')
df_grouped_extra_conservative.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/df_extraConservativeMasking_imageWise.csv')
analyze_treatment_impact(df_grouped_extra_conservative, masking_stringency = 'extra_conservative')
create_kde_plot(df_extra_conservative, masking_stringency = 'extra_conservative')
create_simple_boxplot(df_extra_conservative, masking_stringency = 'extra_conservative')

##### create summary table ########
# List to store results for each scenario
table_data = []

# Compute stats for each scenario
table_data.append(compute_stats(df_baseline, "baseline -- pixel-wise"))
table_data.append(compute_stats(df_conservative, "conservative -- pixel-wise"))
table_data.append(compute_stats(df_extra_conservative, "extra conservative -- pixel-wise"))

table_data.append(compute_stats(df_grouped_baseline, "baseline -- image-wise"))
table_data.append(compute_stats(df_grouped_conservative, "conservative -- image-wise"))
table_data.append(compute_stats(df_grouped_extra_conservative, "extra conservative -- image-wise"))

# Create a dataframe to display the table
columns = [
    'Scenario', 
    'Irrigated Warm Mean Temp', 'Irrigated Cool Mean Temp', 
    'Droughted Warm Mean Temp', 'Droughted Cool Mean Temp', 
    'Difference Warm-Cool Mean (Irrigated)', 'Difference Warm-Cool Mean (Droughted)', 
    'Difference Droughted-Irrigated (Warm)', 'Difference Droughted-Irrigated (Cool)', 
    'p-value (Irrigated)', 'p-value (Droughted)',
    'p-value Treatment (Warm)', 'p-value Treatment (Cool)'  # New columns for p-values for treatment differences
]

df_table = pd.DataFrame(table_data, columns=columns)

# save the table
df_table.to_csv('/Users/bcw269/Documents/Common_gardens/thermal2024/output_sensitivity/sensitivity_output_table.csv')