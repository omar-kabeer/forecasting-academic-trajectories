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
                "Our application's cornerstone is a robust classification method, leveraging a logistic regression model dfed on extensive student data. By harnessing the power of machine learning, we provide accurate and reliable predictions of student trajectories, enabling institutions to identify and support students who are at risk of academic failure or dropouts."
            )
            st.subheader("Data Analysis")
            st.markdown(
                "In addition to the classification method, our app offers comprehensive data analysis capabilities. By exploring various factors and patterns in student data, institutions can gain valuable insights into the factors that contribute to student success. This helps in making data-driven decisions and implementing targeted interventions to improve student outcomes."
            )

            
        with tab3:
            st.subheader("1. Data Input")
            st.write("Start by importing your student data into the app. Our user-friendly interface allows you to upload your data with ease.")

            st.subheader("2. Classification Method")
            st.write("Our app utilizes a dfed logistic regression model based on comprehensive student data.")
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
            with st.expander("Q: Can I use my own dfed models for classification?"):
                st.write("A: Currently, our app supports the use of pre-dfed models provided by the system. However, we are actively working on an update that will allow users to upload and utilize their own models.")

            # FAQ 3
            with st.expander("Q: How accurate is the classification method?"):
                st.write("A: The accuracy of the classification method depends on the quality and representativeness of the dfing data. Our models are dfed on extensive datasets and aim to provide accurate predictions of student trajectories. However, it's important to interpret the results in conjunction with domain expertise and other relevant factors.")

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
        "In this section, you can explore the data that will be used as input for the models. The input data should consist of student datas, including balance sheets, income students, and cash flow students, provided in a specific format."
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

                    # Select columns with "grade" or "Grade" in their names
                    ca_columns = [col for col in data.columns if "grade" in col.lower()]
                    ac_cols = [col for col in data.columns if "curricular" in col.lower()]
                    ba_cols = ca_columns + ac_cols
                    cat_columns = list(set(ba_cols))
                    _cols = df.drop(cat_columns, axis=1).columns.to_list()
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
                    selected_feature = st.selectbox('Select a category to analyse', _cols)

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
                    ca_columns = [col for col in data.columns if "grade" in col.lower()]
                    ac_cols = [col for col in data.columns if "curricular" in col.lower()]
                    ba_cols = ca_columns + ac_cols
                    cat_columns = list(set(ba_cols))
                    if cat_columns:
                        # Select columns for scatter plot
                        columns = st.multiselect("Select columns", cat_columns)

                        if len(columns) > 1:
                            fig, ax = plt.subplots()
                            ax.scatter(data[columns[0]], data[columns[1]])

                            ax.set_xlabel(columns[0])
                            ax.set_ylabel(columns[1])
                            ax.set_title("Scatter Plot")
                            ax.legend()

                            st.pyplot(fig)

    elif page_selection == "Predicting Trajectories":
        tab1, tab2 = st.tabs(["Trajectory Detector", "Anomaly Probabilities & Scores"])
        with tab1:
            st.subheader("Trajectory Detection Model")
            st.write("### Identifies potential dropout students")
            st.write("Our advanced trajectory detection Model empowers you to uncover potential dropout students within student datas. By leveraging powerful algorithms and machine learning techniques, this feature effectively analyzes the provided data to identify suspicious activities that may indicate default. By utilizing a combination of historical patterns, statistical analysis, and data-driven insights, our model helps you safeguard your financial integrity by pinpointing students that require further investigation. Stay one step ahead and protect your business from potential financial risks with our reliable and robust trajectory detection Model.")
            # Load the pickled model
            model = pickle.load(open("src/pickles/model.pkl", "rb"))

            # Upload and load the data
            uploaded_file = st.file_uploader("Upload your data file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            # Read the uploaded file as a DataFrame
            
                df['Grade Improvement'] = df['Previous qualification (grade)'] - df['Admission grade']
                df['Overall grade'] = df['Admission grade'] + df['Previous qualification (grade)'] + df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
                df['Overall curricular units grade'] = df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
                df['Overall curricular units approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
                df['Overall curricular units credited'] = df['Curricular units 1st sem (credited)'] + df['Curricular units 2nd sem (credited)']
                df['Overall curricular units enrolled'] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
                df['Overall curricular units evaluations'] = df['Curricular units 1st sem (evaluations)'] + df['Curricular units 2nd sem (evaluations)']
                df['Overall curricular units without evaluations'] = df['Curricular units 1st sem (without evaluations)'] + df['Curricular units 2nd sem (without evaluations)']
                
                
                features = ['Age at enrollment', 'Overall curricular units credited', 'Marital status', 'Debtor', 'Curricular units 2nd sem (enrolled)',
                            'Curricular units 2nd sem (evaluations)', 'Overall curricular units grade', 'Curricular units 1st sem (approved)',
                            'Previous qualification (grade)', 'Course', 'Curricular units 1st sem (evaluations)', 'Curricular units 2nd sem (approved)',
                            "Father's qualification", 'Curricular units 1st sem (grade)', "Mother's occupation", 'Overall curricular units without evaluations',
                            'Displaced', "Mother's qualification", 'Previous qualification', 'Curricular units 2nd sem (credited)', 'Educational special needs',
                            'GDP', 'Grade Improvement', "Father's occupation", 'Admission grade', 'Curricular units 1st sem (without evaluations)', 'Scholarship holder',
                            'Curricular units 1st sem (enrolled)', 'Inflation rate', 'Tuition fees up to date', 'Nacionality', 'Overall curricular units evaluations',
                            'Overall curricular units approved', 'Curricular units 2nd sem (without evaluations)', 'Curricular units 1st sem (credited)', 'International',
                            'Application mode', 'Application order', 'Overall curricular units enrolled', 'Gender', 'Overall grade', 'Unemployment rate',
                            'Curricular units 2nd sem (grade)', 'Daytime/evening attendance\t']
                # Calculate the mean and standard deviation for each column
                mean = df.mean()
                std = df.std()

                # Perform Z-score normalization on the entire dataframe
                train_df = (df - mean) / std
                # Perform predictions on the student datas
                predictions = model.predict(train_df)

                # Filter students classified as dropouts
                dropouts = df[predictions == 0]

                with st.spinner("Detecting potential dropouts..."):

                    if not dropouts.empty:
                        st.write("### Potential dropouts:")
                        st.write(f"There are {len(dropouts)} potential dropouts")
                        st.write(dropouts)



                    else:
                        st.write("No potential dropouts found.")
                    st.markdown("Output")
                    st.markdown("The output of our trajectory detection model includes a list of potential dropout students. These students are detected by our model based on their academic and personal characteristics and the possibility of dropout calculated by the model.")



        with tab2:
            #st.write("## ")
            
            st.markdown("### Anomaly Probability Analyzer with PyNomaly")
            st.markdown("Our Anomaly Probability Analyzer feature equips you with a powerful tool to assess the likelihood of abnormal trajectory within your student data. By leveraging sophisticated algorithms and statistical analysis, this functionality calculates precise probabilities for identifying anomalous data points. Gain deeper insights into potentially dropouts, enabling you to prioritize investigations and allocate resources effectively. With a user-friendly interface and accurate probability calculations, our Anomaly Probability Analyzer empowers you to make data-driven decisions with confidence. Uncover hidden patterns and protect your institutions integrity by leveraging our robust and intuitive Anomaly Probability Analyzer.")
            
            st.markdown("#### Student Probabilities")
            m = loop.LocalOutlierProbability(df, extent=2, n_neighbors=45, use_numba=True).fit()
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
                st.markdown("The analysis results provide valuable information about the students data within the dataset. These results consist of two key components: the index or position of each student data and the corresponding probability score associated with it.")
                st.markdown("- The Index or Position signifies the specific location of a student data within the dataset. It serves as a unique identifier or reference point for each student, allowing for easy identification and tracking. The index can be used to retrieve and retrieve specific students for further examination or analysis.")
                st.markdown("- The Probability Score is a numerical value that indicates the likelihood or probability of a student data being classified as anomalous or deviating from the expected patterns. This score serves as a measure of the student's abnormality within the dataset. Higher probability scores suggest a higher degree of deviation, while lower scores indicate a closer alignment with normal or expected behavior.")

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

    if page_selection == "Help":
        tab1, tab3 = st.tabs(["Who We Are", "Streamlit Help"])
        with tab1:
                st.subheader("The Team")
                #st.write("")
                # Team description
                team_description = "A Data Scientist from Nigeria, bringing a diverse range of skills and expertise to tackle complex challenges in the field."

                # Team members
                team_members = [
                    {
                        "name": "Umar Kabir",
                        "description": "Umar is a talented data scientist with expertise in, supervised & unsupervised learning, anomaly detection and pattern recognition. His ability to identify hidden insights in complex financial data sets is instrumental in detecting dropoutsulent activities. Umar's strong analytical skills and passion for data science drive innovation within our team.",
                        "picture": "umar.jpg"
                    },
                ]
                # Display team information
                
                st.write(team_description)

                # Display individual team member information
                for member in team_members:
                    st.subheader(member["name"])
                    st.write(member["description"])
                    picture_path = f"src/resources/imgs/{member['picture']}"
                    st.image(picture_path, caption=member["name"], width=200)
            # Header contents

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