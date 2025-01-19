# Purpose of the Study

The purpose of this study is to collect data on apartments for sale in the Antalya province using web scraping techniques. The goal is to analyze the real estate market in various districts of Antalya in detail and provide meaningful insights through visualizations. By uncovering regional price dynamics and trends, the study aims to conduct market valuation and identify profitable investment opportunities.

---

## Data Collection

Data was collected from the **Hepsiemlak** platform using web scraping methods.

### Method 1: Requests and BeautifulSoup
- **HTTP Requests**: Connected to the website using the Requests library.
- **HTML Parsing**: Extracted key data fields using BeautifulSoup.
- **Data Extraction**: Collected data such as price, square meters, number of rooms, and building age.
- **Limitations**: Due to bot protection mechanisms, this method was limited in scope.

### Method 2: Selenium
- **Browser Automation**: Simulated browser behavior using Selenium.
- **Dynamic Waits**: Ensured page elements loaded fully with WebDriverWait.
- **Data Extraction**: Dynamically gathered target data fields.
- **Results**: Data from 252 pages was successfully collected using this method.

---

## Dataset Description

- **Price**: Sale price of the apartment (₺).
- **M2**: Total area of the apartment (in square meters).
- **Number_of_Rooms**: The number of rooms and living spaces.
- **Building_Age**: The age of the building.
- **Floor**: The floor on which the apartment is located.
- **Location**: The neighborhood and district of the apartment.
![image](https://github.com/user-attachments/assets/9d61b5a7-55a9-408e-b5e2-0610f4a21379)

---

## Data Preprocessing
![image](https://github.com/user-attachments/assets/261a9aea-0cd9-4b2d-a0ce-0e6c15dfa9ce)
### Missing Value Imputation
- **M2**: Missing values were filled with the mean.
- **Floor**: Missing values were filled with the mode.
- **Building_Age**: Missing values were predicted using an artificial neural network built with PyTorch.

### Categorical Variable Encoding
- **Label Encoding**: Applied label encoding to all categorical variables.

### Outlier Detection and Handling
![image](https://github.com/user-attachments/assets/623100ad-bb57-4139-b91e-30486da841ac)
- **IQR Method**: Identified and reduced the impact of outliers.

---

## Visualizations

### Box-Plot Graphs
- **Price**: Identified a significant number of outliers.
- **M2**: Broad distribution due to luxury properties.
- **Building_Age**: Detected extreme values caused by data entry errors.

### Violin Plots
- **Price and M2**: Analyzed varying distributions across regions.
- **Building_Age and Number_of_Rooms**: Explored the impact of new and old structures in different regions.

### Distribution Graphs
- **Price and M2**: Concentrated in mid-range properties.
- **Building_Age**: Newer buildings were found to dominate the market.

### Pie Charts
- **Number of Rooms and Floor**: Indicated a preference for mid-sized properties.
- **Regions**: Highlighted Kepez and Muratpaşa as the most popular areas.

---

## Tools and Technologies Used
- **Python**: Requests, BeautifulSoup, Selenium, PyTorch
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
