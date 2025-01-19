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
### Missing Value Imputation
- **M2**: Missing values were filled with the mean.
- **Floor**: Missing values were filled with the mode.
- **Building_Age**: Missing values were predicted using an artificial neural network built with PyTorch.
![image](https://github.com/user-attachments/assets/261a9aea-0cd9-4b2d-a0ce-0e6c15dfa9ce)

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
![image](https://github.com/user-attachments/assets/e81b1ee0-5550-497a-84c5-2336e3468a1a)
![image](https://github.com/user-attachments/assets/45eb4a0c-c95a-488a-b8cd-e97b8a6229a4)
![image](https://github.com/user-attachments/assets/7afbd337-47b8-4c23-bce9-dabc5f77e1a4)

### Violin Plots
- **Price and M2**: Analyzed varying distributions across regions.
- **Building_Age and Number_of_Rooms**: Explored the impact of new and old structures in different regions.
![image](https://github.com/user-attachments/assets/a048cc61-d4a4-4784-9938-4e5dc002fee2)
![image](https://github.com/user-attachments/assets/625be5f7-cc1e-457b-8d31-126cb5854f43)
![image](https://github.com/user-attachments/assets/25dec874-3e84-4171-8086-dda814c53d74)
![image](https://github.com/user-attachments/assets/8ecb5941-1245-4db5-9dc3-707767860af5)

### Distribution Graphs
- **Price and M2**: Concentrated in mid-range properties.
- **Building_Age**: Newer buildings were found to dominate the market.
![image](https://github.com/user-attachments/assets/abcc60a7-b595-4901-924b-7dbf452be4fd)
![image](https://github.com/user-attachments/assets/e49c2a07-1ca9-4c88-b3b9-c929b4f6789f)
![image](https://github.com/user-attachments/assets/13b55ac3-aeb8-4601-80d7-1e47c60a3b94)

### Pie Charts
- **Number of Rooms and Floor**: Indicated a preference for mid-sized properties.
- **Regions**: Highlighted Kepez and Muratpaşa as the most popular areas.
![image](https://github.com/user-attachments/assets/0378c0ea-e2f3-4f87-996e-c20d74d48e63)
![image](https://github.com/user-attachments/assets/57fdee27-30d1-45a8-8595-496864b769fe)
![image](https://github.com/user-attachments/assets/3c2bcfc7-7f16-4a84-9f3a-a56e991badfd)
![image](https://github.com/user-attachments/assets/eb7ad11b-8e4d-4ac2-87df-ea6f4f2ab046)

---

## Tools and Technologies Used
- **Python**: Requests, BeautifulSoup, Selenium, PyTorch
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
