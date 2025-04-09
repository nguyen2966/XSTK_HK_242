# BTL_PD
This project is focused on performing data analysis and visualization using KNN imputation, OLS regression modeling, and statistical description techniques. It is structured in a way that supports modularity, reusability, and clarity.

## **📁 Project Structure **
```
BTL_PD/
├── Chart/ # Output figures generated by the scripts in src/
├── data/ # Input datasets used for analysis
├── Sample_Report/ # Reports or documentation (if any)
├── src/ # Source code for the project
│ ├── KNN_process.py # Script for KNN imputation of missing values
│ ├── OLS_Regression_Model.py # Script for building and evaluating a regression model
│ ├── Thong_Ke_Mo_Ta.py # Script for statistical summaries and visualizations
| ├── 1_and_2_Sample_Test.py # Script to perform test on 1 and 2 sample
| ├── Anova_Model.py # script to generate Anova Model
├── requirements.txt # List of required Python packages
└── Readme.md # Project documentation
```
##
🚀 Getting Started
1. Clone the Repository
git clone <your-repo-url>
cd BTL_PD

2. Install Dependencies
Make sure you have Python 3.7 or higher installed. Then install all required libraries:
```sh
pip install -r requirements.txt
```
3. Prepare Your Data
Place your dataset(s) inside the data/ directory. The scripts are configured to load data from that folder.

4. Run the Scripts
Navigate to the src/ directory and run the desired script:
```sh
cd src
```
```sh
python KNN_process.py
python OLS_Regression_Model.py
python Thong_Ke_Mo_Ta.py
```

5. View Results
All generated plots and charts will be saved in the Chart/ folder automatically.
##

##
📦 Dependencies
The requirements.txt includes all required libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
dataframe-image
scipy
##

📊 Output
Visualizations from the analysis are saved in the Chart/ directory.

Descriptive statistics, regression models, and imputation results are handled by the scripts in src/.

📬 Contact
If you encounter any issues or have suggestions, feel free to create an issue or submit a pull request.