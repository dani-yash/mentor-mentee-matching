# app.py
import streamlit as st
import pandas as pd
import time
import ast
import matchAlgo as ma
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
       
def elems_norm(x):
  lst = []
  for i in x:
    a = i['First_Name']+' '+i['Surname']+' '+str(i['Student_ID'])
    lst.append(a)
  return lst 


def male_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Gender'] == 1:
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def female_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Gender'] == 0:
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def other_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Gender'] == 2:
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def dom_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Domestic_International'] == 'Domestic':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def inter_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Domestic_International'] == 'International':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def prog1_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Academic_Program'] == 'Earth & Space Science & Engineering':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def prog2_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Academic_Program'] == 'Mechanical Engineering':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def prog3_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Academic_Program'] == 'Electrical Engineering & Computer Science':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def prog4_ptc(x):
  lst = []
  size = len(x)
  for i in x:
    if i['Academic_Program'] == 'Civil Engineering':
      lst.append(i)
  ptc = (len(lst)/size)*100
  return int(ptc)


def move_mentee(df, id, clusterFrom, dff, clusterTo):
  for i in df['Elements'][clusterFrom]:
    if int(i['Student_ID']) == int(id):
      df['Elements'][clusterFrom].remove(i)
      dff['Elements'][clusterTo].append(i)


st.set_page_config(page_title='Mentor-Mentee Matching Tool', layout="centered")

# Section 1: App Layout and Navigation
st.title("Mentor-Mentee Matching Tool")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a section:", 
                        [ "Data Input", "Clustering and Matching", "Statistics", "Hobbies"])

st.sidebar.info("This tool helps in forming mentor-mentee clusters based on common availability and traits.")

# Section 2: Data Input - Upload Forms and Cluster Settings
if page == "Data Input":
    st.header("Data Upload and Settings")

    # Upload Mentee Data
    mentee_file = st.file_uploader("Upload Mentee Data (CSV)", type="csv")
    if mentee_file is not None:
        mentee_data = pd.read_csv(mentee_file)
        st.write("Mentee Data Sample:")
        st.write(mentee_data.head())
        mentee_data['Availability'] = mentee_data['Availability'].apply(ast.literal_eval)
        mentee_info = mentee_data[["First_Name", "Surname", "Student_ID", "Email"]]
        mentee_data = mentee_data.drop(columns=["First_Name", "Surname", "Student_ID", "Email"])
        st.session_state['mentee_data'] = mentee_data
        st.session_state['mentee_info'] = mentee_info

    # Upload Mentor Data (first mentor file)
    mentor_file = st.file_uploader("Upload Mentor Data (CSV)", type="csv")
    if mentor_file is not None:
        mentor_data = pd.read_csv(mentor_file)
        st.write("Mentor Data Sample:")
        st.write(mentor_data.head())
        mentor_data['Availability'] = mentor_data['Availability'].apply(ast.literal_eval)
        mentor_info = mentor_data[["First_Name", "Surname", "Student_ID", "Email"]]
        mentor_data = mentor_data.drop(columns=["First_Name", "Surname", "Student_ID", "Email"]) 
        num_clusters = mentor_data['number_of_groups'].sum()
        st.session_state['mentor_data'] = mentor_data
        st.session_state['mentor_info'] = mentor_info
        st.session_state['num_clusters'] = num_clusters

    # Cluster Settings
    st.subheader("Group Settings")
    cluster_min = st.number_input("Min Mentees per Group", min_value=1, max_value=30, value=12,
                                  help="Minimum number of mentees per group.")
    cluster_max = st.number_input("Max Mentees per Group", min_value=1, max_value=30, value=18,
                                  help="Maximum number of mentees per group.")

    st.write("Select the number of mentors per group:")
    option = st.selectbox(
      "How many mentors per group?", 
      ["1", "2"],
      help="Choose 1 or 2 mentors per group."
    )
    mentors_per_group = int(option)

    # If user chooses 2 mentors per group, show second mentor file uploader immediately
    if mentors_per_group > 1:
        st.write("Upload the Second Mentor Data (CSV) for additional mentors:")
        second_mentor_file = st.file_uploader("Upload Second Mentor Data (CSV)", type="csv", key="second_mentor")
        if second_mentor_file is not None:
            second_mentor_data = pd.read_csv(second_mentor_file)
            st.write("Second Mentor Data Sample:")
            st.write(second_mentor_data.head())
            second_mentor_data['Availability'] = second_mentor_data['Availability'].apply(ast.literal_eval)
            second_mentor_info = second_mentor_data[["First_Name", "Surname", "Student_ID", "Email"]]
            second_mentor_data = second_mentor_data.drop(columns=["First_Name", "Surname", "Student_ID", "Email"]) 
            # Merge info with second mentor data
            second_mentor_data = pd.concat([second_mentor_info, second_mentor_data], axis=1)
            st.session_state['second_mentor_data'] = second_mentor_data
        else:
            st.session_state['second_mentor_data'] = None
    else:
        st.session_state['second_mentor_data'] = None

    matchButton = st.button('Match')

    if matchButton:
        # Ensure we have all needed data
        if 'mentee_data' in st.session_state and mentee_file is not None and mentor_file is not None:
            mentee_data = st.session_state['mentee_data']
            mentee_info = st.session_state.get('mentee_info', None)
            mentor_data = st.session_state.get('mentor_data', None)
            mentor_info = st.session_state.get('mentor_info', None)
            num_clusters = st.session_state.get('num_clusters', 40)  # default 40 if not set

            if mentor_data is not None and mentor_info is not None and mentee_info is not None:
                cluster_df, nouns = ma.process_clustering_and_matching(
                    mentee_data, mentor_data, mentee_info, mentor_info, num_clusters, cluster_max, cluster_min
                )

                if mentors_per_group > 1 and st.session_state['second_mentor_data'] is not None:
                    second_mentor_data = st.session_state['second_mentor_data']
                    # Assign second mentors now
                    cluster_df = ma.assign_additional_mentors_from_second_file(cluster_df, second_mentor_data)

                st.session_state['cluster_df'] = cluster_df
                st.session_state['nouns'] = nouns
            else:
                st.warning("Please upload the first mentor file.")
        else:
            st.warning("Please upload both Mentee and Mentor files first.")

# Section 3: Clustering and Matching Display
if page == "Clustering and Matching":
    st.header("Group Display and Manual Adjustments")

    if 'cluster_df' in st.session_state:
        cluster_df = st.session_state['cluster_df']  # Access the cached cluster_df from session_state
        if 'Cluster Size' not in cluster_df.columns:
            cluster_df.insert(1, 'Cluster Size', cluster_df['Elements'].str.len())

        if 'Percentage Male' not in cluster_df.columns:
            cluster_df.insert(2, 'Percentage Male', cluster_df['Elements'].apply(male_ptc))

        if 'Percentage Female' not in cluster_df.columns:
            cluster_df.insert(3, 'Percentage Female', cluster_df['Elements'].apply(female_ptc))

        if 'Percentage Other' not in cluster_df.columns:
            cluster_df.insert(4, 'Percentage Other', cluster_df['Elements'].apply(other_ptc))

        if 'Percentage Domestic' not in cluster_df.columns:
            cluster_df.insert(5, 'Percentage Domestic', cluster_df['Elements'].apply(dom_ptc))

        if 'Percentage International' not in cluster_df.columns:
            cluster_df.insert(6, 'Percentage International', cluster_df['Elements'].apply(inter_ptc))
        cluster_df['Mentees'] = cluster_df['Elements'].apply(elems_norm)

        st.markdown("### Filtering Options")

        if 'min_cluster_size' not in st.session_state:
            st.session_state['min_cluster_size'] = 0
        if 'min_percentage_male' not in st.session_state:
            st.session_state['min_percentage_male'] = 0
        if 'min_percentage_female' not in st.session_state:
            st.session_state['min_percentage_female'] = 0
        if 'min_percentage_other' not in st.session_state:
            st.session_state['min_percentage_other'] = 0
        if 'mentor_name_substring' not in st.session_state:
            st.session_state['mentor_name_substring'] = ""
        if 'availability_substring' not in st.session_state:
            st.session_state['availability_substring'] = ""
        if 'mentee_substring' not in st.session_state:
            st.session_state['mentee_substring'] = ""

        with st.form("filter_form"):
            # Numeric filters
            min_cluster_size = st.number_input("Minimum Cluster Size:", min_value=0, max_value=1000, value=0, help="Show only clusters with at least this number of mentees.")
            min_percentage_male = st.number_input("Min % Male:", min_value=0, max_value=100, value=0, help="Show only clusters with at least this percentage of Male mentees.")
            min_percentage_female = st.number_input("Min % Female:", min_value=0, max_value=100, value=0, help="Show only clusters with at least this percentage of Female mentees.")
            min_percentage_other = st.number_input("Min % Other:", min_value=0, max_value=100, value=0, help="Show only clusters with at least this percentage of Other gender mentees.")
            
            # Text filters (substring searches, case-insensitive)
            mentor_name_substring = st.text_input("Filter by Mentor First Name (contains):", "", help="Show clusters where the assigned mentor's first name contains this substring.")
            availability_substring = st.text_input("Filter by Most Common Availability (contains):", "", help="Show clusters whose most common availability contains this substring.")
            mentee_substring = st.text_input("Filter by Mentee Name/ID (contains):", "", help="Show clusters that have at least one mentee whose info contains this substring.")

            apply_filters_button = st.form_submit_button("Apply Filters")

            if apply_filters_button:
              st.session_state['min_cluster_size'] = min_cluster_size
              st.session_state['min_percentage_male'] = min_percentage_male
              st.session_state['min_percentage_female'] = min_percentage_female
              st.session_state['min_percentage_other'] = min_percentage_other
              st.session_state['mentor_name_substring'] = mentor_name_substring
              st.session_state['availability_substring'] = availability_substring
              st.session_state['mentee_substring'] = mentee_substring

        # Start with the original cluster_df
        filtered_df = cluster_df.copy()

        # Always apply filters based on session_state values
        if st.session_state['min_cluster_size'] > 0:
            filtered_df = filtered_df[filtered_df['Cluster Size'] >= st.session_state['min_cluster_size']]

        if st.session_state['min_percentage_male'] > 0 and 'Percentage Male' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Percentage Male'] >= st.session_state['min_percentage_male']]

        if st.session_state['min_percentage_female'] > 0 and 'Percentage Female' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Percentage Female'] >= st.session_state['min_percentage_female']]

        if st.session_state['min_percentage_other'] > 0 and 'Percentage Other' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Percentage Other'] >= st.session_state['min_percentage_other']]

        if st.session_state['mentor_name_substring'].strip():
            filtered_df = filtered_df[filtered_df['Assigned Mentor First Name'].str.contains(st.session_state['mentor_name_substring'], case=False, na=False)]

        if st.session_state['availability_substring'].strip():
            filtered_df = filtered_df[filtered_df['Most Common Availability'].str.contains(st.session_state['availability_substring'], case=False, na=False)]

        if st.session_state['mentee_substring'].strip():
            lower_substring = st.session_state['mentee_substring'].lower()
            def mentee_matches(m_list):
                return any(lower_substring in s.lower() for s in m_list)
            filtered_df = filtered_df[filtered_df['Mentees'].apply(mentee_matches)]

        # After all filters are applied and before sorting:
        if filtered_df.empty:
            st.warning("No clusters match your filters.")
            # Since no results are available, skip sorting and display steps below:
        else:
            # Proceed with sorting only if filtered_df is not empty
            st.markdown("### Sorting Options")
            sortable_columns = ["Cluster", "Cluster Size", "Percentage Male", "Percentage Female", "Percentage Other", 
                                "Percentage Domestic", "Percentage International"]

            sort_column = st.selectbox("Sort by column:", sortable_columns, help="Select a column to sort the clusters by.")
            sort_ascending = st.radio("Sort order:", ["Ascending", "Descending"], index=0, help="Choose ascending or descending order.")

            if sort_column:
                filtered_df = filtered_df.sort_values(by=sort_column, ascending=(sort_ascending == "Ascending"))

        st.subheader("Grouped Clusters")
        st.dataframe(filtered_df.drop(['Elements'], axis=1),hide_index=True)  # Display the clustered DataFrame

        # Manual Adjustments for Clusters
        adj_cluster_df = cluster_df.copy()

        st.subheader("Manual Adjustments")
        selected_cluster = st.selectbox("Select Cluster to Adjust", adj_cluster_df["Cluster"], index=None, help="Choose which cluster you want to modify.")
        if selected_cluster is not None:
            adj_cluster_df = adj_cluster_df.loc[adj_cluster_df["Cluster"] == selected_cluster]
            st.write(adj_cluster_df)
            gender_display = {0: 'F', 1: 'M', 2: 'O'}
            prog_display = {'Earth & Space Science & Engineering': 'Earth & Space\nEng.', 'Mechanical Engineering': 'Mech. Eng.', 'Electrical Engineering & Computer Science': 'EECS', 'Civil Engineering': 'Civil Eng.'}
            menteesToDisplay = [', '.join([person['First_Name'],person['Surname'],str(person['Student_ID']),gender_display[person['Gender']],prog_display[person['Academic_Program']],', '.join(person['Availability'])]) for person in adj_cluster_df['Elements'][selected_cluster]]
            mentee_to_move = st.selectbox("Select Mentee to Move", menteesToDisplay, index=None, help="Pick a mentee to reassign to a different cluster.")
            if mentee_to_move:
                target_cluster = st.selectbox("Move to Cluster", cluster_df["Cluster"], index=None, help="Select the cluster to which you want to move the chosen mentee.")
                if target_cluster:
                    studentId = mentee_to_move.split(', ')[2]
                    if st.button("Apply Adjustment"):
                        move_mentee(cluster_df,studentId,selected_cluster,cluster_df,target_cluster)
                        st.success(f"Moved {mentee_to_move} to Cluster {target_cluster}.")

                        cluster_df = st.session_state['cluster_df']
                        cluster_df['Cluster Size'] = cluster_df['Elements'].str.len()
                        cluster_df['Percentage Male'] = cluster_df['Elements'].apply(male_ptc)
                        cluster_df['Percentage Female'] = cluster_df['Elements'].apply(female_ptc)
                        cluster_df['Percentage Other'] = cluster_df['Elements'].apply(other_ptc)
                        cluster_df['Percentage Domestic'] = cluster_df['Elements'].apply(dom_ptc)
                        cluster_df['Percentage International'] = cluster_df['Elements'].apply(inter_ptc)
                        cluster_df['Mentees'] = cluster_df['Elements'].apply(elems_norm)
                        time.sleep(1)
                        st.rerun()
    else:
        st.write("No cluster data available. Please run the clustering process first.")

# Section 4: Statistics and Metrics
if page == "Statistics":
    st.header("Mentee Dataset Statistics Overview")
    if 'cluster_df' in st.session_state:
      mentee_data = st.session_state['mentee_data']
      if mentee_data['Gender'].dtype != object:  
       mentee_data['Gender'] = mentee_data['Gender'].map({0: 'Female', 1: 'Male', 2: 'Other'})
      st.write(mentee_data['Gender'].value_counts(normalize=True) * 100)
      st.write(mentee_data['Domestic_International'].value_counts(normalize=True) * 100)
      st.write(mentee_data['Academic_Program'].value_counts(normalize=True) * 100)

      if 'cluster_df' in st.session_state:
          cluster_df = st.session_state['cluster_df']

          st.subheader("Gender Distribution")
          cluster_df_gendist = cluster_df.copy()
          cluster_df_gendist['Male'] = cluster_df_gendist['Elements'].apply(male_ptc)
          cluster_df_gendist['Female'] = cluster_df_gendist['Elements'].apply(female_ptc)
          cluster_df_gendist['Other'] = cluster_df_gendist['Elements'].apply(other_ptc)
          cluster_df_gendist = cluster_df_gendist[['Cluster','Male','Female','Other']].set_index('Cluster')

          fig1 = plt.figure(figsize=(10,6))
          sns.heatmap(cluster_df_gendist,annot=True, fmt="d", cmap="coolwarm", linewidths=0.5)
          st.pyplot(fig1)

          st.subheader("Domestic vs. International")
          cluster_df_studtypdist = cluster_df.copy()
          cluster_df_studtypdist['Domestic'] = cluster_df['Elements'].apply(dom_ptc)
          cluster_df_studtypdist['International'] = cluster_df['Elements'].apply(inter_ptc)
          cluster_df_studtypdist = cluster_df_studtypdist[['Cluster','Domestic','International']].set_index('Cluster')
          fig2 = plt.figure(figsize=(10,6))
          sns.heatmap(cluster_df_studtypdist,annot=True, fmt="d", cmap="coolwarm", linewidths=0.5)
          st.pyplot(fig2)
          
          st.subheader("Academic Programs")
          cluster_df_progdist = cluster_df.copy()
          cluster_df_progdist['Earth & Space\nEng.'] = cluster_df['Elements'].apply(prog1_ptc)
          cluster_df_progdist['Mech. Eng.'] = cluster_df['Elements'].apply(prog2_ptc)
          cluster_df_progdist['EECS'] = cluster_df['Elements'].apply(prog3_ptc)
          cluster_df_progdist['Civil Eng.'] = cluster_df['Elements'].apply(prog4_ptc)
          cluster_df_progdist = cluster_df_progdist[['Cluster','Earth & Space\nEng.','Mech. Eng.', 'EECS', 'Civil Eng.']].set_index('Cluster')

          fig3 = plt.figure(figsize=(10,6))
          sns.heatmap(cluster_df_progdist,annot=True, fmt="d", cmap="coolwarm", linewidths=0.5)
          st.pyplot(fig3)

if page == "Hobbies":
    st.title("Hobbies Word Clouds")

    if 'nouns' in st.session_state:
        nouns = st.session_state['nouns']
        grouped = nouns.groupby('Cluster')

        # Iterate over each cluster group
        for cluster, group in grouped:
            # Combine all hobbies in the cluster into one string
            hobbies_text = " ".join(group['Hobbies'])
            
            # Generate a word cloud for the cluster
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hobbies_text)
            
            # Display the word cloud using Streamlit
            st.subheader(f"Word Cloud for Group {cluster}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')  # Turn off axes
            st.pyplot(fig)
