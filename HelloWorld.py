import matplotlib.pyplot as plt
import statistics
from scipy import stats as st
import numpy as np


# # # ages = [23,23,27,27,39,41, 47, 49, 50, 52, 54, 54, 54, 56, 57, 58, 58, 60, 61, 65]
# # # fat = [9.7, 26.2, 7.5, 17.9, 31.8, 24.9, 27.4, 27.2, 21.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7, 37.4, 36.2]

# # # plt.boxplot(fat)
# # # plt.show()

# # # midpoint interpolation. We have 20 total entries - we take the midpoint of the first 10 for Q1 and the midpoint of the next 10 for Q3.
# # # 

# # # fat.sort()
# # # print(fat)


# # # agesMean = np.mean(ages)
# # # fatMean = np.mean(fat)


# # # print(f"Mean Age: {agesMean:.2f}")
# # # print(f"Mean Body Fat: {fatMean:.2f}%")
# # # print(f"\n")


# # # age1 = np.percentile(ages, 25)
# # # age2 = np.percentile(ages, 50)
# # # age3 = np.percentile(ages, 75)

# # # print(f"First Quartile Age: {age1:.2f}")
# # # print(f"Median Age: {age2:.2f}")
# # # print(f"Third Quartile Age: {age3:.2f}")

# # # fat1 = np.percentile(fat, 25)
# # # fat2 = np.percentile(fat, 50)
# # # fat3 = np.percentile(fat, 75)
# # # print(f"\n")

# # # print(f"First Quartile Fat: {fat1:.2f}%")
# # # print(f"Median Fat: {fat2:.2f}%")
# # # print(f"Third Quartile Fat: {fat3:.2f}%")
# # # print(f"\n")

# # # sdAges = np.std(ages)
# # # print(f"Standard Devation for Age: {sdAges: 2f}")
# # # print(f"\n")

# # # sdFat = np.std(fat)
# # # print(f"Standard Devation for Fat: {sdFat: 2f}")
# # # print(f"\n")

# # # agesMax = np.max(ages)
# # # fatMax = np.max(fat)
# # # agesMin = np.min(ages)
# # # fatMin = np.min(fat)





# # # print(f"Min Age: {agesMin:.2f}")
# # # print(f"Min Fat: {fatMin:.2f}%")

# # # print(f"\n")
# # # print(f"Max Age: {agesMax:.2f}")
# # # print(f"Max Fat: {fatMax:.2f}%")

# # # print(f"\n")
# # # print(st.mode(ages))

# # # fatMode = st.mode(fat)
# # # print(f"Max Fat: {fatMode}%")


# # # ageBins = [1,15,30,45,60,80, 110]
# # # frequencies = [200, 450, 300, 1500, 700, 44]

# # # plt.figure(figsize=(10, 6))
# # # plt.bar(ageBins[:-1], frequencies, width=[14, 15, 15, 15, 20, 30], color='orange', align='edge')
# # # plt.xticks(ageBins)
# # # plt.xlabel('Age')
# # # plt.ylabel('Frequency')
# # # plt.title('Age Distribution')
# # # plt.grid(axis='y')
# # # plt.show()


# # # cumulative_frequencies = []
# # # cumulative = 0
# # # for freq in frequencies:
# # #     cumulative += freq
# # #     cumulative_frequencies.append(cumulative)

# # # total_frequency = cumulative_frequencies[-1]

# # # # Function to find the approximate value for a given percentile
# # # def approximate_percentile(percentile):
# # #     target = percentile * total_frequency
# # #     for i in range(len(cumulative_frequencies)):
# # #         if cumulative_frequencies[i] >= target:
# # #             bin_start = ageBins[i]
# # #             bin_end = ageBins[i + 1]
# # #             bin_freq = frequencies[i]
# # #             bin_cumulative_before = cumulative_frequencies[i - 1] if i > 0 else 0
# # #             within_bin_position = (target - bin_cumulative_before) / bin_freq
# # #             return bin_start + within_bin_position * (bin_end - bin_start)

# # # # Calculate median, Q1, and Q3
# # # median_age = approximate_percentile(0.5)
# # # q1_age = approximate_percentile(0.25)
# # # q3_age = approximate_percentile(0.75)

# # # print(f"Approximate Median Age: {median_age}")
# # # print(f"Approximate Q1 Age: {q1_age}")
# # # print(f"Approximate Q3 Age: {q3_age}")




# # # fig = plt.figure()
# # # fig.suptitle('Ages BoxPlot', fontsize=14, fontweight='bold')

# # # plt.boxplot(ages)
# # # plt.show()
# # # plt.boxplot(ages)
# # # # plt.boxplot(fat)
# # # plt.show()



# # # Data for Monroe and Tippecanoe
# # monroe = [15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 58, 143, 152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 87, 46, 37, 147, 118, 116, 97, 85, 38, 31, 142, 123, 81, 84, 75, 31, 31, 138, 96, 67, 60, 56, 54, 37, 66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 51, 49, 51, 22, 34, 64, 43, 51, 51, 49, 31, 13, 65, 32, 37, 31, 37, 20, 20, 53, 34, 38, 33, 24, 31, 18, 33]

# # tippecanoe = [74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 219, 231, 214, 127, 111, 238, 220, 239, 225, 228, 148, 96, 276, 247, 239, 19, 181, 134, 109, 321, 280, 263, 205, 209, 114, 94, 262, 244, 223, 212, 246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 263, 158, 135, 86, 17, 98, 86, 240, 209, 199, 144, 44, 127, 94, 290, 167, 179, 179, 119, 79, 93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 34, 30, 81, 77, 66, 79, 53, 35, 48, 61]

# # # # Function to calculate Minkowski distance
# # # def minkowski_distance(x, y, h):
# # #     if h == np.inf:
# # #         return np.max(np.abs(np.array(x) - np.array(y)))  # For h = infinity, use max absolute difference
# # #     return np.power(np.sum(np.power(np.abs(np.array(x) - np.array(y)), h)), 1/h)

# # # # Minkowski distance for h = 1 (Manhattan)
# # # distance_h1 = minkowski_distance(monroe, tippecanoe, 1)
# # # print(f'Minkowski Distance (h=1): {distance_h1}')

# # # # Minkowski distance for h = 2 (Euclidean)
# # # distance_h2 = minkowski_distance(monroe, tippecanoe, 2)
# # # print(f'Minkowski Distance (h=2): {distance_h2}')

# # # # Minkowski distance for h = infinity (Chebyshev)
# # # distance_h_inf = minkowski_distance(monroe, tippecanoe, np.inf)
# # # print(f'Minkowski Distance (h=infinity): {distance_h_inf}')

# # dot_product = np.dot(monroe, tippecanoe)
# # print("dot Product:", dot_product)

# import numpy as np

# # # Define the vectors
# # d1 = np.array([15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 
# #                58, 143, 152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 
# #                87, 46, 37, 147, 118, 116, 97, 85, 38, 31, 142, 123, 
# #                81, 84, 75, 31, 31, 138, 96, 67, 60, 56, 54, 37, 
# #                66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 
# #                49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 
# #                51, 49, 51, 22, 34, 64, 43, 51, 51, 49, 31, 
# #                13, 65, 32, 37, 31, 37, 20, 20, 53, 34, 38, 
# #                33, 24, 31, 18, 33])

# # d2 = np.array([74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 
# #                219, 231, 214, 127, 111, 238, 220, 239, 225, 228, 
# #                148, 96, 276, 247, 239, 19, 181, 134, 109, 321, 
# #                280, 263, 205, 209, 114, 94, 262, 244, 223, 212, 
# #                246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 
# #                263, 158, 135, 86, 17, 98, 86, 240, 209, 199, 
# #                144, 44, 127, 94, 290, 167, 179, 179, 119, 79, 
# #                93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 
# #                83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 
# #                34, 30, 81, 77, 66, 79, 53, 35, 48, 61])

# # # Calculate the magnitudes
# # magnitude_d1 = np.linalg.norm(d1)
# # magnitude_d2 = np.linalg.norm(d2)

# # print("Magnitude of d1 (||d1||):", magnitude_d1)
# # print("Magnitude of d2 (||d2||):", magnitude_d2)



# # monroe = [15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 58, 143, 152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 87, 46, 37, 147, 118, 116, 97, 85, 38, 31, 142, 123, 81, 84, 75, 31, 31, 138, 96, 67, 60, 56, 54, 37, 66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 51, 49, 51, 22, 34, 64, 43, 51, 51, 49, 31, 13, 65, 32, 37, 31, 37, 20, 20, 53, 34, 38, 33, 24, 31, 18, 33]
# # tippecanoe = [74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 219, 231, 214, 127, 111, 238, 220, 239, 225, 228, 148, 96, 276, 247, 239, 19, 181, 134, 109, 321, 280, 263, 205, 209, 114, 94, 262, 244, 223, 212, 246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 263, 158, 135, 86, 17, 98, 86, 240, 209, 199, 144, 44, 127, 94, 290, 167, 179, 179, 119, 79, 93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 34, 30, 81, 77, 66, 79, 53, 35, 48, 61]
# # months = ['November 2020', 'December 2020', 'January 2021']

# # bar_width = 0.5

# # x = 3
# # plt.bar(x - bar_width/2, monroe, width=bar_width, label='Monroe County, IN')
# # plt.bar(x + bar_width/2, tippecanoe, width=bar_width, label='Tippecanoe County, IN')

# # plt.title('The Number of COVID-19 Positive Cases in Monroe County, IN and Tippecanoe County, IN from November 2020 until January 2021')
# # plt.xlabel('Number of COVID-19 Positive Cases')
# # plt.ylabel('Month')

# # plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# Provided data
monroe = [
    15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 58, 143, 
    152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 87, 46, 37, 147, 118, 
    116, 97, 85, 38, 31, 142, 123, 81, 84, 75, 31, 31, 138, 96, 67, 
    60, 56, 54, 37, 66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 
    49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 51, 49, 51, 22, 
    34, 64, 43, 51, 51, 49, 31, 13, 65, 32, 37, 31, 37, 20, 20, 
    53, 34, 38, 33, 24, 31, 18, 33
]

tippecanoe = [
    74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 219, 231, 214, 
    127, 111, 238, 220, 239, 225, 228, 148, 96, 276, 247, 239, 19, 
    181, 134, 109, 321, 280, 263, 205, 209, 114, 94, 262, 244, 223, 
    212, 246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 263, 158, 
    135, 86, 17, 98, 86, 240, 209, 199, 144, 44, 127, 94, 290, 167, 
    179, 179, 119, 79, 93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 
    83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 34, 30, 81, 77, 66, 
    79, 53, 35, 48, 61
]

# Assuming the last three values correspond to November, December, January
# Calculate the totals for the last 3 months
# monroe_totals = [sum(monroe[-30:]), sum(monroe[-60:-30]), sum(monroe[-90:-60])]
# tippecanoe_totals = [sum(tippecanoe[-30:]), sum(tippecanoe[-60:-30]), sum(tippecanoe[-90:-60])]

# # Month labels
# months = ['November 2020', 'December 2020', 'January 2021']

# # Bar width
# bar_width = 0.35

# # Set the positions of the bars
# x = np.arange(len(months))

# # Create the bar chart
# plt.bar(x - bar_width/2, monroe_totals, width=bar_width, label='Monroe County, IN', color='blue')
# plt.bar(x + bar_width/2, tippecanoe_totals, width=bar_width, label='Tippecanoe County, IN', color='orange')

# # Add titles and labels
# plt.title('The Number of COVID-19 Positive Cases in Monroe County, IN and Tippecanoe County, IN from November 2020 until January 2021')
# plt.xlabel('Months')
# plt.ylabel('Number of COVID-19 Positive Cases')

# # Set the ticks and labels
# plt.xticks(x, months)


# # Show the plot
# plt.tight_layout()
# plt.show()


# # from mpmath import mp, mpf, quad, exp, power
# # mp.dps = 100

# # def chi_square_pdf(x, k):
# #     return power(x, (k/2) - 1) * exp(-x/2) / (power(2, k/2) * mp.gamma(k/2))

# # chi_square_statistic = mpf('1824.032')
# # degrees_of_freedom = 1
# # p_value = quad(lambda x: chi_square_pdf(x, degrees_of_freedom), [chi_square_statistic, mp.inf])

# # print(f"The p-value with 100 digits of precision is:\n{p_value}")




import numpy as np
import matplotlib.pyplot as plt

# monroe = [
#     15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 58, 143, 
#     152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 87, 46, 37, 147, 118, 
#     116, 97, 85, 38, 31, 142, 123, 81, 84, 75, 31, 31, 138, 96, 67, 
#     60, 56, 54, 37, 66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 
#     49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 51, 49, 51, 22, 
#     34, 64, 43, 51, 51, 49, 31, 13, 65, 32, 37, 31, 37, 20, 20, 
#     53, 34, 38, 33, 24, 31, 18, 33
# ]

# tippecanoe = [
#     74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 219, 231, 214, 
#     127, 111, 238, 220, 239, 225, 228, 148, 96, 276, 247, 239, 19, 
#     181, 134, 109, 321, 280, 263, 205, 209, 114, 94, 262, 244, 223, 
#     212, 246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 263, 158, 
#     135, 86, 17, 98, 86, 240, 209, 199, 144, 44, 127, 94, 290, 167, 
#     179, 179, 119, 79, 93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 
#     83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 34, 30, 81, 77, 66, 
#     79, 53, 35, 48, 61
# ]

# # Assuming the last three values correspond to November, December, January
# # Calculate the totals for the last 3 months
# monroe_totals = [sum(monroe[-30:]), sum(monroe[-60:-30]), sum(monroe[-90:-60])]
# tippecanoe_totals = [sum(tippecanoe[-30:]), sum(tippecanoe[-60:-30]), sum(tippecanoe[-90:-60])]

# # Month labels
# months = ['November 2020', 'December 2020', 'January 2021']

# # Bar width
# bar_width = 0.35

# # Set the positions of the bars
# y = np.arange(len(months))

# # Create the bar chart with flipped axes
# plt.barh(y - bar_width/2, monroe_totals, height=bar_width, label='Monroe County, IN', color='blue')
# plt.barh(y + bar_width/2, tippecanoe_totals, height=bar_width, label='Tippecanoe County, IN', color='orange')

# # Add titles and labels
# plt.title('The Number of COVID-19 Positive Cases in Monroe County, IN and Tippecanoe County, IN from November 2020 until January 2021')
# plt.ylabel('Months')
# plt.xlabel('Number of COVID-19 Positive Cases')

# # Set the ticks and labels
# plt.yticks(y, months)

# # Add a legend
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# monroe = [
#     15, 67, 60, 84, 61, 95, 20, 23, 110, 117, 91, 97, 105, 46, 58, 143, 
#     152, 153, 139, 69, 32, 13, 106, 112, 106, 10, 87, 46, 37, 147, 118, 
#     116, 97, 85, 38, 31, 142, 123, 81, 84, 75, 31, 31, 138, 96, 67, 
#     60, 56, 54, 37, 66, 77, 40, 37, 4, 44, 42, 97, 72, 82, 54, 30, 
#     49, 32, 138, 79, 76, 72, 66, 44, 34, 100, 55, 51, 49, 51, 22, 
#     34, 64, 43, 51, 51, 49, 31, 13, 65, 32, 37, 31, 37, 20, 20, 
#     53, 34, 38, 33, 24, 31, 18, 33
# ]

# tippecanoe = [
#     74, 126, 173, 216, 213, 160, 100, 93, 259, 247, 219, 231, 214, 
#     127, 111, 238, 220, 239, 225, 228, 148, 96, 276, 247, 239, 19, 
#     181, 134, 109, 321, 280, 263, 205, 209, 114, 94, 262, 244, 223, 
#     212, 246, 115, 85, 250, 224, 159, 168, 182, 109, 70, 263, 158, 
#     135, 86, 17, 98, 86, 240, 209, 199, 144, 44, 127, 94, 290, 167, 
#     179, 179, 119, 79, 93, 175, 124, 119, 134, 94, 55, 47, 118, 92, 
#     83, 86, 79, 36, 37, 87, 78, 74, 72, 51, 34, 30, 81, 77, 66, 
#     79, 53, 35, 48, 61
# ]

# # Calculate the totals for the last 3 months
# monroe_totals = [sum(monroe[-30:]), sum(monroe[-60:-30]), sum(monroe[-90:-60])]
# tippecanoe_totals = [sum(tippecanoe[-30:]), sum(tippecanoe[-60:-30]), sum(tippecanoe[-90:-60])]

# # Month labels
# months = ['November 2020', 'December 2020', 'January 2021']

# # Bar width
# bar_width = 0.35

# # Set the positions of the bars
# y = np.arange(len(months))

# # Create the bar chart with flipped axes and order of bars
# plt.barh(y, monroe_totals, height=bar_width, label='Monroe County, IN', color='blue')
# plt.barh(y + bar_width, tippecanoe_totals, height=bar_width, label='Tippecanoe County, IN', color='orange')

# # Add titles and labels
# plt.title('The Number of COVID-19 Positive Cases in Monroe County, IN and Tippecanoe County, IN from November 2020 until January 2021')
# plt.ylabel('Months')
# plt.xlabel('Number of COVID-19 Positive Cases')

# # Set the ticks and labels
# plt.yticks(y + bar_width / 2, months)  # Center the month labels between the two bars

# # Add a legend
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the CSV file
# data = pd.read_csv('covid-data-county.csv', index_col=0)

# # Calculate total cases for each month for both counties
# monroe_nov = data.loc['Monroe', '11/1/20':'11/30/20'].sum()
# monroe_dec = data.loc['Monroe', '12/1/20':'12/31/20'].sum()
# monroe_jan = data.loc['Monroe', '1/1/21':'1/31/21'].sum()

# tippecanoe_nov = data.loc['Tippecanoe', '11/1/20':'11/30/20'].sum()
# tippecanoe_dec = data.loc['Tippecanoe', '12/1/20':'12/31/20'].sum()
# tippecanoe_jan = data.loc['Tippecanoe', '1/1/21':'1/31/21'].sum()

# # Data for plotting
# months = ['November 2020', 'December 2020', 'January 2021']
# monroe_cases = [monroe_nov, monroe_dec, monroe_jan]
# tippecanoe_cases = [tippecanoe_nov, tippecanoe_dec, tippecanoe_jan]

# # Create the bar chart
# fig, ax = plt.subplots(figsize=(12, 6))

# # Set the positions of the bars on the y-axis
# y_pos = np.arange(len(months))

# # Create bars
# ax.barh(y_pos - 0.2, monroe_cases, 0.4, label='Monroe County', color='skyblue')
# ax.barh(y_pos + 0.2, tippecanoe_cases, 0.4, label='Tippecanoe County', color='orange')

# # Customize the chart
# ax.set_yticks(y_pos)
# ax.set_yticklabels(months)
# ax.invert_yaxis()  # Labels read top-to-bottom
# ax.set_xlabel('Number of Positive COVID-19 Cases')
# ax.set_title('The Number of COVID-19 Positive Cases in Monroe County, IN and\nTippecanoe County, IN from November 2020 until January 2021')
# ax.legend()

# # Add value labels on the bars
# for i, v in enumerate(monroe_cases):
#     ax.text(v, i - 0.2, str(v), va='center', fontweight='bold')
# for i, v in enumerate(tippecanoe_cases):
#     ax.text(v, i + 0.2, str(v), va='center', fontweight='bold')

# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # Load the CSV file
# data = pd.read_csv('covid-data-county.csv', index_col=0)

# # Calculate total cases for each month for both counties
# monroe_nov = data.loc['Monroe', '11/1/20':'11/30/20'].sum()
# monroe_dec = data.loc['Monroe', '12/1/20':'12/31/20'].sum()
# monroe_jan = data.loc['Monroe', '1/1/21':'1/31/21'].sum()

# tippecanoe_nov = data.loc['Tippecanoe', '11/1/20':'11/30/20'].sum()
# tippecanoe_dec = data.loc['Tippecanoe', '12/1/20':'12/31/20'].sum()
# tippecanoe_jan = data.loc['Tippecanoe', '1/1/21':'1/31/21'].sum()

# # Data for plotting
# months = ['November 2020', 'December 2020', 'January 2021']
# monroe_cases = [monroe_nov, monroe_dec, monroe_jan]
# tippecanoe_cases = [tippecanoe_nov, tippecanoe_dec, tippecanoe_jan]

# # Create the bar chart
# fig, ax = plt.subplots(figsize=(12, 6))

# # Set the positions of the bars on the y-axis
# y_pos = np.arange(len(months))

# # Create bars
# ax.barh(y_pos - 0.2, monroe_cases, 0.4, label='Monroe County', color='skyblue')
# ax.barh(y_pos + 0.2, tippecanoe_cases, 0.4, label='Tippecanoe County', color='orange')

# # Customize the chart
# ax.set_yticks(y_pos)
# ax.set_yticklabels(months)
# # Remove invert_yaxis() to have months in ascending order
# ax.set_xlabel('Number of Positive COVID-19 Cases')
# ax.set_title('The Number of COVID-19 Positive Cases in Monroe County, IN and\nTippecanoe County, IN from November 2020 until January 2021')
# ax.legend()

# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv('covid-data-county.csv', index_col=0)

# Calculate total cases for each month for both counties
monroe_nov = data.loc['Monroe', '11/1/20':'11/30/20'].sum()
monroe_dec = data.loc['Monroe', '12/1/20':'12/31/20'].sum()
monroe_jan = data.loc['Monroe', '1/1/21':'1/31/21'].sum()

tippecanoe_nov = data.loc['Tippecanoe', '11/1/20':'11/30/20'].sum()
tippecanoe_dec = data.loc['Tippecanoe', '12/1/20':'12/31/20'].sum()
tippecanoe_jan = data.loc['Tippecanoe', '1/1/21':'1/31/21'].sum()

# Data for plotting
months = ['November 2020', 'December 2020', 'January 2021']
monroe_cases = [monroe_nov, monroe_dec, monroe_jan]
tippecanoe_cases = [tippecanoe_nov, tippecanoe_dec, tippecanoe_jan]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Set the positions of the bars on the y-axis
y_pos = np.arange(len(months))

# Create bars (Tippecanoe first, then Monroe on top)
ax.barh(y_pos - 0.2, tippecanoe_cases, 0.4, label='Tippecanoe County', color='orange')
ax.barh(y_pos + 0.2, monroe_cases, 0.4, label='Monroe County', color='skyblue')

# Customize the chart
ax.set_yticks(y_pos)
ax.set_yticklabels(months)
ax.set_xlabel('Number of Positive COVID-19 Cases')
ax.set_title('The Number of COVID-19 Positive Cases in Monroe County, IN and\nTippecanoe County, IN from November 2020 until January 2021')
ax.legend()

plt.tight_layout()
plt.show()