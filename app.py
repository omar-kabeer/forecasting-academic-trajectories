import numpy as np
# Data handling dependencies
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tempfile
#import wkhtmltopdf
from io import BytesIO
import base64
from PyNomaly import loop
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.pca import PCA as ppca
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
#from pyod.models.xgbod import XGBOD
from pyod.models.gmm import GMM
# Streamlit dependencies
import streamlit as st
import benford as bf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
#from streamlit_extras.colored_header import colored_header
#from streamlit_extras.dataframe_explorer import dataframe_explorer
#from streamlit_extras.mention import mention
#from streamlit_extras.metric_cards import style_metric_cards
#from streamlit_extras.no_default_selectbox import selectbox
# st.set_page.config(layout='wide', initial_sidebar_state='expanded')
from streamlit_option_menu import option_menu
from streamlit_shap import st_shap
st.set_option('deprecation.showPyplotGlobalUse', False)


# App declaration
def main():
    df = pd.read_csv("src/data/test.csv")
    #df = df.drop(['Target', 'Unnamed: 0'], axis=1)
    from PIL import Image
    # Load image
    image2 = Image.open('src/resources/imgs/1.jpg')

    # Display image in the sidebar
    st.sidebar.image(image2, caption='Academic Success')

    # page options
    pages = {
        "Home": [],
        "Explore Your Data": [],
        "Predicting Trajectories": [],
        "Help": []
    }
    #col1 = st.expander("Page Selection")
    # Add widgets and content to the columns
    #col1.selectbox("Select Page", list(pages.keys()))
    

    # Display dropdown menu in the sidebar for page selection
    page_selection = st.sidebar.selectbox("Select Page", list(pages.keys()))
    

    # Display selected page content
    if page_selection == "Home":
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "About", "How it Works", "Frequently Asked Questions (FAQs)"])
        with tab1:
            # Header contents
            st.title('Predicting Student Trajectories')
            st.subheader("Mitigating Risk, Detecting, and Preventing Failure in Academics")
            st.markdown(
                "Welcome to our advanced Streamlit application for Predicting Students Academic Trajectories. Designed with the utmost precision, our app offers a comprehensive suite of features to empower institutions, NGOs, and interested parties in mitigating potential failure and ensuring the success of their students."
            )

            st.image('src/resources/imgs/2.jpg')
            st.write("### Advantages of Using Machine Learning for Predicting Student Success")
            st.markdown("- Superior Predictive Power: Machine learning models excel at analyzing various factors and patterns in student data to predict their likelihood of success. These models can uncover complex relationships that may not be apparent through traditional methods.")
            st.markdown("- Handling Complex Data: Machine learning algorithms can handle diverse types of student data, such as demographics, academic records, socio-economic factors, and more. By integrating and analyzing these data sources, ML models can provide holistic insights into student success.")
            st.markdown("- Personalized Interventions: ML models can help identify students who are at risk of academic failure or dropouts. This enables institutions to intervene early and provide personalized support, improving the chances of student success.")

            
                
        with tab2:
            st.markdown("With an intuitive and user-friendly interface, our Streamlit application brings together the power of machine learning and statistical analysis in a single platform. Seamlessly navigate through features, analyze data, and gain actionable insights to effectively predict student success and make informed decisions in the education sector.")
            st.markdown("Join us on the forefront of leveraging machine learning for predicting student trajectories. Experience the unparalleled capabilities of our Streamlit app and unlock new levels of risk management, data analysis, and informed decision-making in the education industry.")
            st.subheader("Our Products & Services")
            
            st.subheader('Classification Method')
            st.markdown(
                "Our application's cornerstone is a robust classification method, leveraging a logistic regression model trained on extensive student data. By harnessing the power of machine learning, we provide accurate and reliable predictions of student trajectories, enabling institutions to identify and support students who are at risk of academic failure or dropouts."
            )
            st.subheader("Data Analysis")
            st.markdown(
                "In addition to the classification method, our app offers comprehensive data analysis capabilities. By exploring various factors and patterns in student data, institutions can gain valuable insights into the factors that contribute to student success. This helps in making data-driven decisions and implementing targeted interventions to improve student outcomes."
            )

            
        with tab3:
            st.subheader("1. Data Input")
            st.write("Start by importing your student data into the app. Our user-friendly interface allows you to upload your data with ease.")

            st.subheader("2. Classification Method")
            st.write("Our app utilizes a trained logistic regression model based on comprehensive student data.")
            st.write("The classification method employs sophisticated machine learning techniques to accurately predict student trajectories and assess their likelihood of success in your dataset.")
            st.write("Gain insights into the risk profiles of your students, enabling you to make informed decisions and allocate resources effectively.")

            st.subheader("3. Anomaly Detection")
            st.write("Our app incorporates advanced anomaly detection techniques to identify and isolate unusual patterns or behaviors within student data.")
            st.write("Discover potential outliers and identify students who may require additional support or interventions to ensure their success.")
            st.write("Early detection of anomalies allows you to take proactive measures to mitigate potential risks and support student achievement.")

            st.subheader("4. Probability Estimation")
            st.write("With the integration of statistical methods, our app provides estimated probabilities for each student's likelihood of success.")
            st.write("Prioritize your efforts based on these probabilities, focusing on students who may need additional support and optimizing your resources effectively.")

            st.subheader("5. Comprehensive Analysis")
            st.write("Our app leverages a range of analytical techniques to provide a comprehensive analysis of student trajectories.")
            st.write("Benefit from a versatile set of tools specifically designed for analyzing student data, enabling you to identify factors that contribute to student success.")
            st.write("Detect potential areas of improvement, strengthen support strategies, and ensure the success of your students.")

            st.subheader("6. Actionable Insights")
            st.write("Through our user-friendly interface, explore and visualize the results of the classification method, Benford's analysis, and anomaly detection techniques.")
            st.write("Gain actionable insights and make informed decisions to effectively predict student success and ensure positive academic trajectories.")


        with tab4:
            # FAQ 1
            with st.expander("Q: How do I import my student data into the app?"):
                st.write("A: To import your student data, click on the 'Upload Data' button on the page. Follow the instructions to select and upload your file.")

            # FAQ 2
            with st.expander("Q: Can I use my own trained models for classification?"):
                st.write("A: Currently, our app supports the use of pre-trained models provided by the system. However, we are actively working on an update that will allow users to upload and utilize their own models.")

            # FAQ 3
            with st.expander("Q: How accurate is the classification method?"):
                st.write("A: The accuracy of the classification method depends on the quality and representativeness of the training data. Our models are trained on extensive datasets and aim to provide accurate predictions of student trajectories. However, it's important to interpret the results in conjunction with domain expertise and other relevant factors.")

            # FAQ 4
            with st.expander("Q: Can I export the results of the analysis?"):
                st.write("A: Yes, our app provides an option to export the results. You can save the classification outcomes, anomaly detection results, and other relevant insights in a format of your choice for further analysis or reporting purposes.")

            # FAQ 5
            with st.expander("Q: Is my data secure?"):
                st.write("A: We prioritize data security and privacy. Your uploaded data is processed securely and is not shared with any third parties. We adhere to strict security protocols and comply with relevant data protection regulations.")

            # FAQ 6
            with st.expander("Q: How often should I update the data in the app?"):
                st.write("A: It is recommended to update your data regularly to ensure that the classification models and anomaly detection algorithms remain effective. The frequency of updates may depend on factors such as changes in the student population, curriculum updates, or shifts in the educational landscape.")

            # FAQ 7
            with st.expander("Q: Can I customize the anomaly detection algorithms used?"):
                st.write("A: At the moment, our app offers a set of predefined anomaly detection algorithms. However, we are continuously working to enhance our app's capabilities and plan to introduce customization options in future updates.")


    # Display selected page content
    elif page_selection == "Explore Your Data":
        st.markdown(
        "In this section, you can explore the data that will be used as input for the models. The input data should consist of financial statements, including balance sheets, income statements, and cash flow statements, provided in a specific format."
    )
        st.markdown(
        "The page validates the data entered by you and provides feedback in case of any errors or inconsistencies."
    )
        # Upload and load the data
        uploaded_file = st.file_uploader("Upload your data file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        
            
            tabs1, tabs2 = st.tabs(["Show raw data", "Explore data"])
            with tabs1:
                
                #df = df.drop('Unnamed: 0', axis=1)

                st.subheader("Students Info")
                st.write('Here is an overview of your data')
                st.dataframe(data.head(5))  # Same as st.write(df)
                st.write(f"There are {data.shape[0]} students information in your Data")
                st.write(f"There are {data.shape[1]} columns in your Data")
                #st.subheader("Summary statistics of the dataset")
                st.markdown("### Summary Statistics of the Dataset")
                st.markdown("Summary statistics provides a concise overview of the main characteristics of your dataset.")

                st.markdown("To compute summary statistics, you can use various statistical measures, including:")
                st.markdown("- **Count**: The number of non-missing values in each column.")
                st.markdown("- **Mean**: The average value of each numerical column.")
                st.markdown("- **std(Standard Deviation)**: A measure of the dispersion or variability in each numerical column.")
                st.markdown("- **Min**: The minimum value in each column.")
                st.markdown("- **25%(Percentile)**: The value below which 25% of the data falls in each column.")
                st.markdown("- **50%(Median)**: The middle value in each column, also known as the 50th percentile or the second quartile.")
                st.markdown("- **75%(Percentile)**: The value below which 75% of the data falls in each column.")
                st.markdown("- **Max**: The maximum value in each column.")


                st.markdown("Feel free to explore the summary statistics of your own dataset to gain a better understanding "
                            "of its characteristics and distributions.")
                st.write(data.describe())
            with tabs2:
                sel1, sel2 = st.tabs(["Visualise Your Data", "Statistical Analysis"])
                with sel1:

                    # Get numerical columns from the filtered DataFrame
                    #df = df.drop(['FinancialsDate', 'Date'], axis=1)
                    cat_columns = data.columns.tolist()
                    #print(cat_columns)
                    # Add a default option for the numerical feature selectbox
                    cat_columns = cat_columns
                    st.markdown("### Visualizing the Data")
                    st.markdown("When analyzing categorical data, bar charts are a common visualization choice. They represent "
                                "the distribution of categories within the data.")

                    st.markdown("To create a bar chart that shows the percentage of each category within the data:")
                    st.markdown("Select a categorical variable of interest.")
                    st.markdown("- The app will group the data by the selected variable and calculate the count of each category.")
                    st.markdown("- Then calculate the percentage of each category by dividing the count by the total number of data points.")
                    st.markdown("- A bar chart with the categories on the y-axis and the percentage on the x-axis will be displayed.")
                    st.markdown("Feel free to explore the categorical data in your own dataset and create meaningful bar charts "
                                "to gain insights and communicate your findings effectively.")

                    # Display the numerical feature selectbox
                    selected_feature = st.selectbox('Select a category to analyse', cat_columns)

                    # Check if a numerical feature is selected
                    if selected_feature != 'Select a category to analyse':

                        st.warning('The graphs that appear are interactive and can be zoomed in using the expand button on the top right corner of the graph')
                        counts = data[selected_feature].value_counts()
                        percentages = counts / counts.sum() * 100
                        _df = pd.concat([counts, percentages], axis=1)
                        _df.columns = ['Count', 'Percentage']
                        _df = _df.sort_values(by='Percentage', ascending=False)
                        _df = _df[_df['Count'] >= 0]

                        sns.set(style="darkgrid")
                        plt.figure(figsize=(10, 5))

                        # create the bar chart
                        ax = sns.barplot(y=_df.index, x="Percentage", orient = 'h', data=_df)

                        # set the chart title and axis labels
                        plt.title("Percentage of Students by " + selected_feature)
                        plt.ylabel(selected_feature)
                        plt.xlabel("Percentage")
                        for i in ax.containers:
                            ax.bar_label(i, label_type='edge', fontsize=8, rotation='horizontal', fmt='%.2f%%')

                        # rotate the x-axis labels
                        plt.xticks(rotation='horizontal')
                        st.pyplot()
                with sel2:
                    st.markdown("### Statistical analysis for a Selected Feature")
                    st.markdown("Statistical analysiss are useful for visualizing trends and patterns in data over time.")

                    st.markdown("To create a Statistical analysis that shows the changes in a selected feature for a specific category "
                                "and time, follow these steps:")
                    st.markdown("1. Select a category of interest.")
                    st.markdown("2. Select a feature to analyze.")

                    st.markdown("Feel free to explore the data in your own dataset and create meaningful plots "
                                "to analyze trends, or other temporal patterns.")


                    # Select columns with "grade" or "Grade" in their names
                    cat_columns = [col for col in data.columns if "grade" in col.lower()]
                    if cat_columns:
                        # Select columns for scatter plot
                        columns = st.multiselect("Select columns", cat_columns)

                    #print(cat_columns)
                    # Add a default option for the numerical feature selectbox
                    #cat_columns = ['Select a category to analyse'] + cat_columns

                        ax.scatter(data[columns[0]], data[columns[1]])

                        ax.set_xlabel(cat_columns)
                        ax.set_ylabel("Values")
                        ax.set_title("Scatter Plot")
                        ax.legend()

                        st.pyplot()

    elif page_selection == "Credit Risk Modelling":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fraud Detector", "Benford's Analysis", "Outlier Detector", "Anomaly Probabilities & Scores", "PyOD Outlier Detector"])
        with tab1:
            st.subheader("Fraud Detection Model")
            st.write("### Identifies potential defaulting companies")
            st.write("Our advanced Fraud Detection Model empowers you to uncover potential defaulting companies within financial statements. By leveraging powerful algorithms and machine learning techniques, this feature effectively analyzes the provided data to identify suspicious activities that may indicate default. By utilizing a combination of historical patterns, statistical analysis, and data-driven insights, our model helps you safeguard your financial integrity by pinpointing statements that require further investigation. Stay one step ahead and protect your business from potential financial risks with our reliable and robust Fraud Detection Model.")
            # Load the pickled model
            model = pickle.load(open("resources/pickles/RandomForestClassifier.pkl", "rb"))

            
            # Read the uploaded file as a DataFrame
            
            df = df2.drop('Unnamed: 0', axis=1)
            # Perform predictions on the financial statements
            predictions = model.predict(df)

            # Filter statements classified as defaulters
            defaulters = df[predictions == 1]

            with st.spinner("Detecting potential fraud..."):

                if not defaulters.empty:
                    st.write("### Potential Defaulters:")
                    st.write(f"There are {len(defaulters)} potential defaulters")
                    st.write(defaulters)



                else:
                    st.write("No potential defaulters found.")
                st.markdown("Output")
                st.markdown("The output of our fraud detection model includes a list of potential defaulting companies. These companies are flagged by our trained model based on their financial statement characteristics and the possibility of default calculated by the model.")

        with tab2:
            st.write("### Statistical Anomaly Detection")
            st.write("#### Detects unusual patterns in financial data")
            st.write("Our cutting-edge Benford's Analysis feature empowers you to detect unusual patterns in financial data with precision and accuracy. By leveraging statistical techniques and the principles of Benford's Law, this powerful tool examines the distribution of leading digits in numerical data to identify potential anomalies. Uncover irregularities and uncover hidden patterns that may signify fraudulent or suspicious activities in your financial statements. With our user-friendly interface and robust analytical capabilities, you can confidently navigate your financial data and make informed decisions to protect your organization's integrity and mitigate risks. Detect and address anomalies effectively with our reliable Benford's Analysis feature.")
            
            df = df.drop(['ID', 'DimFacilityKey', 'Unnamed: 0', 'Default', 'Year', 'Month', 'Week', 'Day', 'FinancialsDate'], axis=1)


            tab1, tab2 = st.tabs(["Benford's Analysis", "Industry Benford's Analysis"])
            with tab1:
                # Get numerical columns from the filtered DataFrame
                numerical_columns = df.select_dtypes(include='number').columns.tolist()
                # Add a default option for the numerical feature selectbox
                numerical_columns = ['Select a financial feature to analyse'] + numerical_columns

                # Display the numerical feature selectbox
                selected_feature = st.selectbox('Select a financial feature to analyse', numerical_columns)

                # Check if a numerical feature is selected
                if selected_feature != 'Select a financial feature to analyse':
                    # Perform Benford's Law analysis on the selected feature
                    feature_data = df[selected_feature].dropna()

                    benf = bf.Benford(feature_data, confidence=95)
                    report = benf.F1D.report(high_Z='all')
                    table = benf.F1D.T
                    mad = benf.F1D.MAD
                    mads = [0.006, 0.012, 0.015]
                    st.write("## Benford's Law Analysis for", selected_feature)
                    st.markdown(f"The table below shows the expected number of times each first digit should appear in the dataset, as well as the actual number of times each first digit appears in {selected_feature} of the dataset.")
                    st.markdown("- The 'Expected' row is based on Benford's Law expected frequencies")
                    st.markdown("- The 'Found' row shows the frequencies found in the dataset, while the 'Counts' column is the actual number of times each first digit appeared in the dataset.")
                    st.markdown("- The 'Dif' row shows the difference between the expected and actual number of times each first digit appeared.")
                    st.markdown("- The 'AbsDif' row shows the absolute value of the difference between the expected and actual number of times each first digit appeared.")
                    st.markdown("- The 'Z_score' row shows how many standard deviations away from the expected number of times each first digit appeared.")
                    st.markdown("In general, a Z-score of 2 or more is considered to be a significant deviation from the expected number of times a first digit should appear. If the Z_score is above 2 it could be a sign of fraud, as fraudsters often manipulate financial data to make it appear more legitimate.")
                    # Display the text report
                    st.write(table)


                    # Plot the graph from the report
                    #benf.plot(title='Benford Distribution')
                    st.pyplot(report)

                    st.markdown("The Mean Absolute Deviation (MAD) helps us understand how spread out the scores are. It tells us on average how far each score is from the average score.")
                    #mads = MAD_CONFORM[digs]
                    if mad <= mads[0]:
                        st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                        st.write(f'Close conformity.\n')
                    elif mad <= mads[1]:
                        st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                        st.write(f'Acceptable conformity.\n')
                    elif mad <= mads[2]:
                        st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                        st.write(f'Marginally Acceptable conformity.\n')
                    else:
                        st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                        st.write(f'Nonconformity.\n')
                    st.write("It is important to note that Benford's Law is not a perfect indicator of fraud. There are many other factors that can contribute to a deviation from Benford's Law, such as rounding errors, data entry errors, and the natural variation of financial data. However, Benford's Law can be a useful tool for identifying potential fraud, and it should be used in conjunction with other fraud detection techniques.")

            with tab2:
                st.markdown('''Selecting the industry is a crucial step in effectively detecting financial statement fraud. Different industries have unique characteristics, data patterns, and risk profiles. By choosing the relevant industry, you can apply specialized fraud detection models and analysis techniques tailored to that specific sector, increasing the accuracy and effectiveness of your fraud detection efforts.

Each industry may have its own set of common fraudulent activities, transaction types, or financial indicators that require focused attention. By selecting the appropriate industry, you can leverage industry-specific expertise and knowledge to identify potential anomalies and uncover fraudulent behaviors more efficiently.

Furthermore, industry-specific analysis allows you to benchmark against industry norms and identify outliers or irregularities that may indicate fraudulent activities specific to that sector. This targeted approach enhances the precision of your fraud detection efforts and enables you to take appropriate actions promptly.

Selecting the industry ensures that your financial statement fraud detection app aligns with the unique requirements and risks associated with the specific industry, enabling you to proactively safeguard your organization's financial integrity.

'''
)

                # Get unique industry values from the 'Industry' column
                industries = df['Industry'].unique()

                # Add a default option for the selectbox
                industries = ['Select an industry'] + list(industries)

                # Display the selectbox
                selected_industry = st.selectbox('Select an industry', industries)

                # Check if an industry is selected
                if selected_industry != 'Select an industry':
                    # Filter the DataFrame based on the selected industry
                    filtered_df = df[df['Industry'] == selected_industry]

                    # Get numerical columns from the filtered DataFrame
                    numerical_columns = filtered_df.select_dtypes(include='number').columns.tolist()

                    # Add a default option for the numerical feature selectbox
                    numerical_columns = ['Select a financial feature to analyse'] + numerical_columns

                    # Display the numerical feature selectbox
                    selected_feature = st.selectbox('Select a financial feature', numerical_columns)

                    # Check if a numerical feature is selected
                    if selected_feature != 'Select a financial feature to analyse':
                        # Perform Benford's Law analysis on the selected feature
                        feature_data = filtered_df[selected_feature].dropna()

                        benf = bf.Benford(feature_data, confidence=95)
                        report = benf.F1D.report(high_Z='all')
                        table = benf.F1D.T
                        mad = benf.F1D.MAD
                        mads = [0.006, 0.012, 0.015]
                        st.write("## Benford's Law Analysis for", selected_feature)
                        st.markdown(f"The table below shows the expected number of times each first digit should appear in the dataset, as well as the actual number of times each first digit appears in {selected_feature} of the dataset.")
                        st.markdown("- The 'Expected' row is based on Benford's Law expected frequencies")
                        st.markdown("- The 'Found' row shows the frequencies found in the dataset, while the 'Counts' column is the actual number of times each first digit appeared in the dataset.")
                        st.markdown("- The 'Dif' row shows the difference between the expected and actual number of times each first digit appeared.")
                        st.markdown("- The 'AbsDif' row shows the absolute value of the difference between the expected and actual number of times each first digit appeared.")
                        st.markdown("- The 'Z_score' row shows how many standard deviations away from the expected number of times each first digit appeared.")
                        st.markdown("In general, a Z-score of 2 or more is considered to be a significant deviation from the expected number of times a first digit should appear. If the Z_score is above 2 it could be a sign of fraud, as fraudsters often manipulate financial data to make it appear more legitimate.")
                        # Display the text report
                        st.write(table)


                        # Plot the graph from the report
                        #benf.plot(title='Benford Distribution')
                        st.pyplot(report)

                        st.markdown("The Mean Absolute Deviation (MAD) helps us understand how spread out the scores are. It tells us on average how far each score is from the average score.")
                        #mads = MAD_CONFORM[digs]
                        if mad <= mads[0]:
                            st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                            st.write(f'Close conformity.\n')
                        elif mad <= mads[1]:
                            st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                            st.write(f'Acceptable conformity.\n')
                        elif mad <= mads[2]:
                            st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                            st.write(f'Marginally Acceptable conformity.\n')
                        else:
                            st.markdown(f"#### The Mean Absolute Deviation for {selected_feature} is {round(mad,3)}")
                            st.write(f'Nonconformity.\n')
                        st.write("It is important to note that Benford's Law is not a perfect indicator of fraud. There are many other factors that can contribute to a deviation from Benford's Law, such as rounding errors, data entry errors, and the natural variation of financial data. However, Benford's Law can be a useful tool for identifying potential fraud, and it should be used in conjunction with other fraud detection techniques.")
                            
                            
        with tab3:
            #st.write("## Outlier Detector")
            st.write("### Flags unusual or anomalous financial statements")
            st.markdown("Our advanced Outlier Detector feature provides a reliable means to identify unusual or anomalous data points within your financial statements. By employing our powerful algorithm the **Isolation Forest**, this tool effectively flags data points that deviate significantly from the expected patterns or trends. Uncover hidden irregularities, potential errors, or even fraudulent activities that may impact your financial integrity. With a user-friendly interface and accurate detection capabilities, our Outlier Detector empowers you to make informed decisions and take proactive measures to address anomalies promptly. Safeguard your financial data with confidence using our robust and efficient Outlier Detector.")
            
            df = df.drop(
                ['FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date', 'ReturnEquityRatio'], axis=1)

            # Encode categorical columns using one-hot encoding
            encoder = OneHotEncoder()
            encoded_cat_columns = encoder.fit_transform(df[['Financial_Type', 'Country', 'Industry']])
            encoded_cat_columns_df = pd.DataFrame(encoded_cat_columns.toarray(),
                                                  columns=encoder.get_feature_names_out(
                                                      ['Financial_Type', 'Country', 'Industry']))

            # Combine encoded categorical columns with numerical columns
            X = pd.concat(
                [df.drop(['Financial_Type', 'Country', 'Industry', 'Unnamed: 0', 'Default'], axis=1), encoded_cat_columns_df],
                axis=1)

            iforest = IsolationForest(contamination=0.03, max_samples='auto', bootstrap=False, n_jobs=-1, random_state=42)
            iforest_ = iforest.fit(X)
            y_pred = iforest_.predict(X)

            y_score = iforest.decision_function(X)
            neg_value_indices = np.where(y_score < 0)
            # Filter statements classified as defaulters
            anomalies = df[y_score < 0]

            if not anomalies.empty:
                st.write("### Anomalous statements:")
                st.write(f"There are {len(anomalies)} anomalies in your data")
                st.write(anomalies)


            if st.checkbox("Global Interpretability"):
                st.subheader("Global Machine Learning Interpretability")
                st.markdown("The global interpretability section presents a summary plot and a bar plot, which offer "
                            "valuable insights into the overall patterns and characteristics of the identified "
                            "anomalies in your financial statements.")
                st.markdown("This information is valuable because it allows us to focus on the most relevant features when investigating anomalies. By understanding which aspects of the financial statements are contributing the most to the model's decision, we can make more informed decisions and take appropriate actions.")

                exp = shap.TreeExplainer(iforest) #Explainer
                shap_values = exp.shap_values(X)  #Calculate SHAP values
                shap.initjs()

                tb1, tb2, tb3 = st.tabs(["Summary Plot", "Bar Plot", "Force Plot"])
                with tb1:
                    st.markdown("SHAP (SHapley Additive exPlanations) is a method used in machine learning to explain the predictions made by a model. In our case, the model is analyzing financial statements and identifying anomalies. The SHAP summary plot shows us which features have the most impact on the model's decision.")
                    summary_plot = shap.summary_plot(shap_values, X)
                    st.pyplot(summary_plot)
                    st.markdown("Think of features as different pieces of information in the financial statements. For example, revenue, expenses, profitability ratios, and other financial metrics could be features. The SHAP summary plot ranks these features based on their importance in detecting anomalies.")

                with tb2:
                    st.markdown("The plot displays the features on the y-axis, with the most important feature at the top and the least important at the bottom. Each feature is represented by a horizontal bar. The length of the bar indicates the magnitude of the feature's impact. A longer bar means that the feature has a larger influence in identifying anomalies.")
                    summary_plot = shap.summary_plot(shap_values, X,plot_type="bar")
                    st.pyplot(summary_plot)
                    st.markdown("So, by looking at the SHAP summary plot, we can quickly see which features play a significant role in identifying anomalies in the financial statements. It helps us understand the key factors driving the detection of unusual patterns.")

                with tb3:
                    st.title("Force Plot")
                    st.markdown("A force plot is a visualization technique that helps us understand how individual features or factors contribute to a specific prediction made by a model. In our case, we use force plots to analyze financial statements and understand the factors that contribute to identifying anomalies or unusual patterns.")

                    # Add a selectbox to choose the index
                    index = st.selectbox("Select an index", range(len(df)))

                    # Display the force plot for the selected index
                    force_plot = shap.force_plot(exp.expected_value, shap_values[index], features=X.iloc[index], feature_names=X.columns)
                    st_shap(force_plot, height=200, width=800)
                    st.markdown("By examining the force plot, we can see which features have the most significant impact on the model's decision for a specific company's financial statement. It helps us understand the specific factors that contribute to identifying anomalies or unusual patterns.")


                    
        with tab4:
            #st.write("## ")
            
            st.markdown("### Anomaly Probability Analyzer with PyNomaly")
            st.markdown("Our Anomaly Probability Analyzer feature equips you with a powerful tool to assess the likelihood of abnormal data within your financial statements. By leveraging sophisticated algorithms and statistical analysis, this functionality calculates precise probabilities for identifying anomalous data points. Gain deeper insights into potentially fraudulent or suspicious activities, enabling you to prioritize investigations and allocate resources effectively. With a user-friendly interface and accurate probability calculations, our Anomaly Probability Analyzer empowers you to make data-driven decisions with confidence. Uncover hidden risks and protect your financial integrity by leveraging our robust and intuitive Anomaly Probability Analyzer.")
            
            #df = df.drop(['ID', 'Default', 'Unnamed: 0', 'DimFacilityKey', 'FinancialsDate', 'Year', 'Month', 'Week', 'Day', 'Date', 'ReturnEquityRatio'], axis=1)
            tab1, tab2 = st.tabs(["Fraud Probabilities", "Industry Fraud Probabilities"])
            with tab1:


                # Encode categorical columns using one-hot encoding
                encoder = OneHotEncoder()
                encoded_cat_columns = encoder.fit_transform(df[['Financial_Type', 'Country', 'Industry']])
                encoded_cat_columns_df = pd.DataFrame(encoded_cat_columns.toarray(), columns=encoder.get_feature_names_out(
                    ['Financial_Type', 'Country', 'Industry']))

                # Combine encoded categorical columns with numerical columns
                X = pd.concat([df.drop(['Financial_Type', 'Country', 'Industry'], axis=1), encoded_cat_columns_df], axis=1)
                # Create an LOF instance

                m = loop.LocalOutlierProbability(X, extent=2, n_neighbors=45, use_numba=True).fit()
                scores1 = m.local_outlier_probabilities

                df['pynomaly_probabilities'] = scores1

                col1, col2 = st.tabs(["Probabilities", "Scores"])

                with col1:
                    st.subheader('Probability Range')
                    #st.write(df['pynomaly_probabilities'])

                    prob_min = st.slider("Minimum Probability", float(df['pynomaly_probabilities'].min()), float(df['pynomaly_probabilities'].max()))
                    prob_max = st.slider("Maximum Probability", float(df['pynomaly_probabilities'].min()), float(df['pynomaly_probabilities'].max()), value=float(df['pynomaly_probabilities'].max()))
                    filtered_probs = df[(df['pynomaly_probabilities'] >= prob_min) & (df['pynomaly_probabilities'] <= prob_max)]
                    st.write(filtered_probs['pynomaly_probabilities'])
                    st.markdown("The analysis results provide valuable information about the financial statements within the dataset. These results consist of two key components: the index or position of each financial statement and the corresponding probability score associated with it.")
                    st.markdown("- The Index or Position signifies the specific location of a financial statement within the dataset. It serves as a unique identifier or reference point for each statement, allowing for easy identification and tracking. The index can be used to retrieve and retrieve specific statements for further examination or analysis.")
                    st.markdown("- The Probability Score is a numerical value that indicates the likelihood or probability of a financial statement being classified as anomalous or deviating from the expected patterns. This score serves as a measure of the statement's abnormality within the dataset. Higher probability scores suggest a higher degree of deviation, while lower scores indicate a closer alignment with normal or expected behavior.")

                with col2:
                    st.subheader('Scores')
                    df['score'] = pd.cut(df['pynomaly_probabilities'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=[1, 2, 3, 4, 5], right=False)
                    score_options = ['All', 1, 2, 3, 4, 5]
                    selected_score = st.selectbox('Select Score', score_options)
                    if selected_score != 'All':
                        filtered_scores = df[df['score'] == selected_score]
                        st.write(filtered_scores['score'])
                    else:
                        st.write(df['score'])

            with tab2:
                st.markdown('''Selecting the industry is a crucial step in effectively detecting financial statement fraud. Different industries have unique characteristics, data patterns, and risk profiles. By choosing the relevant industry, you can apply specialized fraud detection models and analysis techniques tailored to that specific sector, increasing the accuracy and effectiveness of your fraud detection efforts.

Each industry may have its own set of common fraudulent activities, transaction types, or financial indicators that require focused attention. By selecting the appropriate industry, you can leverage industry-specific expertise and knowledge to identify potential anomalies and uncover fraudulent behaviors more efficiently.

Furthermore, industry-specific analysis allows you to benchmark against industry norms and identify outliers or irregularities that may indicate fraudulent activities specific to that sector. This targeted approach enhances the precision of your fraud detection efforts and enables you to take appropriate actions promptly.

Selecting the industry ensures that your financial statement fraud detection app aligns with the unique requirements and risks associated with the specific industry, enabling you to proactively safeguard your organization's financial integrity.

'''
)
                # Get unique industry values from the 'Industry' column
                industries = df['Industry'].unique()

                # Add a default option for the selectbox
                industries = ['Select an industry to detect'] + list(industries)

                # Display the selectbox
                selected_industry = st.selectbox('Select an industry to detect', industries)

                # Check if an industry is selected
                if selected_industry != 'Select an industry to detect':
                    # Filter the DataFrame based on the selected industry
                    filtered_df = df[df['Industry'] == selected_industry]
                    filtered_df = filtered_df.drop(['Financial_Type', 'Country', 'Industry'], axis=1)

                    m = loop.LocalOutlierProbability(filtered_df, extent=2, n_neighbors=45, use_numba=True).fit()
                    scores1 = m.local_outlier_probabilities

                    filtered_df['pynomaly_probabilities'] = scores1

                    # Display 'Probabilities' and 'Scoring' sections side by side
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader('Probabilities')
                        st.write(filtered_df['pynomaly_probabilities'])

                    with col2:
                        st.subheader('Scoring')
                        filtered_df['score'] = pd.cut(filtered_df['pynomaly_probabilities'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=[1, 2, 3, 4, 5], right=False)
                        st.write(filtered_df['score'])

                            
        with tab5:
            st.markdown("### PyOD Suspicious Activity Detector")
            st.markdown("Our Suspicious Activity Detector feature is designed to identify potentially fraudulent or suspicious behavior within your financial statements. This powerful tool utilizes advanced algorithms and data science techniques to flag transactions and activities that raise red flags. By analyzing patterns, anomalies, and risk indicators, our detector helps you uncover and investigate potential instances of financial misconduct or fraudulent behavior. With a user-friendly interface and accurate detection capabilities, our Suspicious Activity Detector empowers you to safeguard your organization's financial integrity and take prompt action against suspicious activities. Stay vigilant and protect your business from potential financial risks with our reliable Suspicious Activity Detector.")
            
            df = df.drop(['Unnamed: 0', 'Default'], axis=1)

            df = df.drop(['Industry', 'Country', 'Financial_Type', 'ID', 'DimFacilityKey'], axis=1)
            scaler = StandardScaler()
            predictions = []
            # Creating a dictionary to store the outlier detection models
            models = {
                #'AutoEncoder': AutoEncoder(contamination=0.03),
                'PCA': ppca(contamination=0.03),
                'IForest': IForest(contamination=0.03),
                'MinimumCovarianceDeterminant': MCD(contamination=0.03),
                'FeatureBagging': FeatureBagging(contamination=0.03),
                }

            # Training each model and storing their anomaly scores
            with st.spinner("Detecting anomalies..."):
                for name, model in models.items():
                    X_scaled = scaler.fit_transform(df)
                    model.fit(X_scaled)
                    prediction = model.predict(X_scaled)
                    predictions.append(prediction)

                    # Hide epoch information
                    #st.balloons()
            #autoencoder_pred = list(predictions[0])
            IForest_pred = list(predictions[1])

            pca_pred = list(predictions[0])

            mcd_pred = list(predictions[2])

            fb_pred = list(predictions[3])


            pyod_preds = pd.DataFrame()
            #pyod_preds['autoencoder_pred'] = autoencoder_pred
            pyod_preds['if_pred'] = IForest_pred
            pyod_preds['pca_pred'] = pca_pred

            pyod_preds['mcd_pred'] = mcd_pred

            pyod_preds['fb_pred'] = fb_pred
            # Calculate the mean prediction for each financial statement
            pyod_preds['mean_prediction'] = pyod_preds.mean(axis=1)
            # Filter statements classified as defaulters
            predicts = df[pyod_preds['mean_prediction'] >= 0.5]

            if not predicts.empty:
                st.write("### Potential Anomalies:")
                st.write(f"There are {len(predicts)} potential anomalies")
                st.write(predicts)

                    
    if page_selection == "Help":
        tab1, tab2, tab3 = st.tabs(["Who We Are", "Resources", "Streamlit Help"])
        with tab1:
            ab1, ab2 = st.tabs(["ExploreAI", "Our Team"])
            with ab1:
                st.subheader("About ExploreAI")
                st.write("ExploreAI builds AI-driven software and digital twins for global companies. We are proud of domain expertise in the utilities, insurance, banking, and telecommunications industries."
                         "We are able to help you accelerate your digital teams: train your workforce, hire talent, or sponsor students through a data science programme. These offerings are powered by the ExploreAI Academy: a learning institution teaching data and AI skills for the next generation."
                         "We have offices in London, Cape Town, Johannesburg, Durban, and Mauritius. We consult to clients in the UK, the US, the Nordics, and South Africa.")
            with ab2:
                #st.subheader("Financial Statement Fraud Detection Team 6")
                
                # Team description
                team_description = "We are a team of data science interns at ExploreAI, working on financial statement fraud detection. Our team consists of talented individuals from South Africa and Nigeria, bringing a diverse range of skills and expertise to tackle complex challenges in the field."

                # Team members
                team_members = [
                    {
                        "name": "Ngawethu Mtirara",
                        "description": "Ngawethu  Mtirara is Junior data scientist, with a background in Electrical Engineering.  With a cup of coffee in hand, Ngawethu's energy knows no bounds, ready to conquer any data challenge that comes his way. Armed with a passion for discovering patterns and extracting insights, Ngawethu fearlessly dives into projects, fueled by the invigorating power of caffeine. Beyond the realm of data, Ngawethu's love for animals is boundless, finding inspiration in the wonders of the natural world.",
                        "picture": "ngawethu.jpg"
                    },
                    {
                        "name": "Idongesit Bokeime",
                        "description": "Idongesit is a detail-oriented data scientist with a strong background in the banking industry with 8 years experience as head of operations. With her analytical mindset and deep understanding of the finance industry, she played a crucial role in uncovering patterns. Idongesit's commitment to accuracy and precision ensures the reliability of our fraud detection models.",
                        "picture": "idongesit.jpg"
                    },
                    {
                        "name": "Manoko Langa",
                        "description": "Manoko Langa identifies as a highly motivated Junior data scientist with a strong background in the natural sciences field. During his studies in the natural sciences field, he recognized the incredible potential of data analysis and because of this strong interest that he's developed towards data analysis, he decided to pivot his career towards data science by enrolling with ExploreAI Academy where he learned advanced skills in programming languages such as Python, R, SQL, statistical analysis, machine learning, artificial intelligence and data visualization. With a blend of such technical skills and his outstanding communication and collaboration skills, it is clear that he could not have chosen any better career path.",
                        "picture": "manoko.jpg"
                    },
                    {
                        "name": "Umar Kabir",
                        "description": "Umar is a talented data scientist with expertise in anomaly detection and pattern recognition. His ability to identify hidden insights in complex financial data sets is instrumental in detecting fraudulent activities. Umar's strong analytical skills and passion for data science drive innovation within our team.",
                        "picture": "umar.jpg"
                    },
                    {
                        "name": "Kayode Gideon Oloyede",
                        "description": "Kayode is a skilled data scientist with good analytical skills. Kayode has good attention to detail and critical thinking abilities.",
                        "picture": "kayode.jpg"
                    }
                ]

                # Display team information
                st.subheader("Financial Statement Fraud Detection Team 6")
                st.write(team_description)

                # Display individual team member information
                for member in team_members:
                    st.subheader(member["name"])
                    st.write(member["description"])
                    picture_path = f"resources/imgs/{member['picture']}"
                    st.image(picture_path, caption=member["name"], width=200)
            # Header contents
        with tab2:
            st.header("This page contains some useful resources")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    "[Financial Statemets](https://www.investopedia.com/terms/f/financial-statements.asp) [![Investopedia](https://icons.iconarchive.com/icons/papirus-team/papirus-apps/16/notion-icon.png)](https://www.investopedia.com/terms/f/financial-statements.asp)"
                )

            with col2:
                st.markdown(
                    "[Benford's Law](https://statisticsbyjim.com/probability/benfords-law/) [![External Link](https://icons.iconarchive.com/icons/iconsmind/outline/16/External-Link-icon.png)](https://statisticsbyjim.com/probability/benfords-law/)"
                )

            with col3:
                st.markdown(
                    "[White paper](https://drive.google.com/file/d/16sAqHxtkHZiP53LFs1rdrTV2qsFj0MFW/view?usp=sharing) [![PDF](https://icons.iconarchive.com/icons/paomedia/small-n-flat/16/file-pdf-icon.png)](https://drive.google.com/file/d/16sAqHxtkHZiP53LFs1rdrTV2qsFj0MFW/view?usp=sharing)"
                )

            with col1:
                st.markdown(
                    "[Benford-py GitHub](https://github.com/milcent/benford_py) [![GitHub](https://icons.iconarchive.com/icons/limav/flat-gradient-social/16/Github-icon.png)](https://github.com/milcent/benford_py)"
                )

            with col2:
                st.markdown(
                    "[PyOD GitHub](https://github.com/yzhao062/pyod) [![GitHub](https://icons.iconarchive.com/icons/limav/flat-gradient-social/16/Github-icon.png)](https://github.com/yzhao062/pyod)"
                )

            with col3:
                st.markdown(
                    "[PyNomaly GitHub](https://github.com/vc1492a/PyNomaly) [![GitHub](https://icons.iconarchive.com/icons/limav/flat-gradient-social/16/Github-icon.png)](https://github.com/vc1492a/PyNomaly)"
                )

            
                

        with tab3:
            st.header("Welcome to the Streamlit Help Page")

            st.subheader("Getting Started")
            st.write("If you are new to Streamlit, here are a few steps to get you started:")
            st.markdown("- Install Streamlit: You can install Streamlit by running `pip install streamlit` in your terminal.")
            st.markdown("- Create a Python file: Create a new Python file and import the Streamlit library using `import streamlit as st`.")
            st.markdown("- Write your app: Write your app code using Streamlit's easy-to-use APIs.")
            st.markdown("- Run the app: In your terminal, navigate to the directory containing your Python file and run `streamlit run your_app.py`.")
            st.markdown("- Interact with your app: Streamlit will provide a local URL where you can view and interact with your app in the browser.")

            st.subheader("Documentation and Resources")
            st.write("To learn more about Streamlit, refer to the following resources:")
            st.markdown("- Streamlit Documentation: Visit the [Streamlit documentation](https://docs.streamlit.io/) for detailed information, guides, and examples.")
            st.markdown("- Streamlit Gallery: Explore the [Streamlit Gallery](https://streamlit.io/gallery) to discover a wide range of example apps and use cases.")
            st.markdown("- Streamlit GitHub Repository: Check out the [Streamlit GitHub repository](https://github.com/streamlit/streamlit) for source code, issue tracking, and community discussions.")

            st.subheader("Community Support")
            st.write("If you have any questions, need assistance, or want to connect with the Streamlit community, here are some helpful resources:")
            st.markdown("- Streamlit Community Forum: Join the [Streamlit Community Forum](https://discuss.streamlit.io/) to ask questions, share ideas, and get help from the community.")
            st.markdown("- Stack Overflow: Use the [Stack Overflow](https://stackoverflow.com/) platform to search for existing answers or ask new questions using the 'streamlit' tag.")
            st.markdown("- Streamlit on Twitter: Follow [@streamlit](https://twitter.com/streamlit) on Twitter to stay updated with the latest Streamlit news, updates, and announcements.")

                
if __name__ == '__main__':
    main()        