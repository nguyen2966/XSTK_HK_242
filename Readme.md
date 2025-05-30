# BTL_PD
This project is focused on performing data analysis and visualization using KNN imputation, OLS regression modeling, and statistical description techniques. It is structured in a way that supports modularity, reusability, and clarity.

## **📁 Project Structure **
```
BTL_PD/
├── Chart/ # Output figures generated by the scripts in src/
├── data/ # Input datasets used for analysis
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
```sh
git clone https://github.com/nguyen2966/XSTK_HK_242.git
cd XSTK_HK_242
```

2. Install Dependencies
Make sure you have Python 3.7 or higher installed. Then install all required libraries:
```sh
pip install -r requirements.txt
```
3. Run the Scripts
Navigate to the src/ directory and run the desired script:
```sh
cd src
```
```sh
python KNN_process.py
python Descriptive_Statistic.py
python 1_and_2_Samples_Test.py
python Anova_Model.py
python OLS_Regression_Model.py
```
4. View Results
All generated plots and charts will be saved in the Chart/ folder automatically.
##
📦 Dependencies: 
The requirements.txt includes all required libraries:
```sh
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
dataframe-image
scipy
```
##

📊 Output
Visualizations from the analysis are saved in the Chart/ directory.

Descriptive statistics, regression models, and imputation results are handled by the scripts in src/.

📬 Contact
If you encounter any issues or have suggestions, feel free to create an issue or submit a pull request.
