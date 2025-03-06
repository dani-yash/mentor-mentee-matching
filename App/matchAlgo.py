import os
import pandas as pd
import random
os.system("python -m spacy download en_core_web_md")
import spacy
nlp = spacy.load("en_core_web_md")

from spacy.tokens import Doc
import numpy as np
from sklearn.neighbors import kneighbors_graph

# pip install numpy pandas scikit-learn matplotlib

import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min

from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict



##############################################
############ Supporting functions ############
##############################################



def text_to_doc(text):
    """
    Converts a given text into a spaCy Doc object.

    Parameters:
        text (str): The input text to process.

    Returns:
        Doc: A spaCy Doc object representing the processed text.
    """    
    return nlp(text)


def text_to_vector(text):
    """
    Converts a given text into a vector representation using spaCy.

    Parameters:
        text (str): The input text to convert.

    Returns:
        numpy.ndarray: A vector representation of the text.
    """
    doc = nlp(text)
    return doc.vector


def compute_similarity(doc1, doc2):
    """
    Computes the similarity between two spaCy Doc objects.

    Parameters:
        doc1 (Doc): The first spaCy Doc object.
        doc2 (Doc): The second spaCy Doc object.

    Returns:
        float: A similarity score between 0 and 1, where higher scores indicate greater similarity.
    """
    return doc1.similarity(doc2)


def extract_nouns_or_verbs(text):
    """
    Extracts either nouns or verbs from a given text, excluding certain common words.

    If nouns are present, the function returns them as a string. 
    If no nouns are found, it falls back to extracting verbs. 
    If neither nouns nor verbs are found, the original text is returned.

    Parameters:
        text (str): The input text from which to extract nouns or verbs.

    Returns:
        str: A space-separated string of lemmatized nouns or verbs, or the original text if none are found.
    """
    
    doc = nlp(text)
    excluded_words = {'hobby', 'hobbies', 'time', 'weekend', 'have', 'thing', 'lot', 'interest', 'spot', 'group', 'activity', 'topic', 'style', 'place', 'life', 'come', 'like', 'love'}
    nouns = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.lemma_ not in excluded_words]
    if nouns:
        return ' '.join(nouns)
    verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
    return ' '.join(verbs) if verbs else text


class Text_KMeans:
    """
    Implements K-Means clustering for text data using spaCy Doc vectors.

    Attributes:
        n_clusters (int): The number of clusters to form.
        max_iter (int): The maximum number of iterations to run the algorithm.
        tol (float): The tolerance level to determine convergence.

    Methods:
        fit(documents):
            Fits the model to the provided text data.
        predict(documents):
            Predicts the cluster labels for new text data.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        """
        Initializes the Text_KMeans clustering model with specified parameters.

        Parameters:
            n_clusters (int): Number of clusters to form. Default is 3.
            max_iter (int): Maximum number of iterations. Default is 100.
            tol (float): Tolerance to declare convergence. Default is 1e-4.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, documents):
        """
        Fits the K-Means model to the provided text data.

        Parameters:
            documents (list of str): List of input text documents.
        """
        docs = [text_to_doc(doc) for doc in documents]

        # Ensure we do not pick more clusters than documents
        if len(docs) < self.n_clusters:
            self.n_clusters = max(1, len(docs))  # At least one cluster, or a fraction if you prefer


        random_idx = np.random.choice(len(docs), self.n_clusters, replace=False)
        centroids = [docs[idx] for idx in random_idx]

        for iteration in range(self.max_iter):



            labels = []
            for doc in docs:
                similarities = [compute_similarity(doc, centroid) for centroid in centroids]
                labels.append(np.argmax(similarities))


            new_centroids = []
            for i in range(self.n_clusters):
                cluster_docs = [docs[j] for j in range(len(docs)) if labels[j] == i]

                if cluster_docs:

                    cluster_vectors = np.array([doc.vector for doc in cluster_docs])
                    avg_vector = np.mean(cluster_vectors, axis=0)


                    new_centroid = Doc(nlp.vocab, words=[])
                    new_centroid.vector = avg_vector
                    new_centroids.append(new_centroid)
                else:

                    new_centroids.append(centroids[i])


            centroid_movements = np.array([
                compute_similarity(centroids[i], new_centroids[i]) for i in range(self.n_clusters)
            ])


            if np.all(centroid_movements < self.tol):
                break

            centroids = new_centroids

        self.labels_ = labels
        self.cluster_centers_ = centroids

    def predict(self, documents):
        """
        Predicts the cluster labels for new text documents.

        Parameters:
            documents (list of str): List of input text documents.

        Returns:
            list of int: Cluster labels for the input documents.
        """
        docs = [text_to_doc(doc) for doc in documents]
        labels = []


        for doc in docs:
            similarities = [compute_similarity(doc, centroid) for centroid in self.cluster_centers_]
            labels.append(np.argmax(similarities))

        return labels



def create_combined_affinity(availability_data, feature_data):
    """
    Creates a combined affinity matrix based on availability and feature data.

    Parameters:
        availability_data (pd.DataFrame): Binary availability matrix where each cell indicates if a participant is available at a given time.
        feature_data (pd.DataFrame): Feature data used to compute similarity between participants.

    Returns:
        np.ndarray: Combined affinity matrix where positive values indicate similarity and negative values indicate dissimilarity.
    """

    n = availability_data.shape[0]
    affinity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                if np.sum(availability_data.iloc[i, :] * availability_data.iloc[j, :]) > 0:
                    affinity_matrix[i][j] = 1
                else:
                    affinity_matrix[i][j] = -1000


    feature_affinity = 1 - pairwise_distances(feature_data, metric='euclidean')

    combined_affinity = feature_affinity + affinity_matrix
    combined_affinity[combined_affinity < 0] = 0

    return combined_affinity


def find_matching_clusters(row, cluster_dict):
    """
Finds clusters that match a participant's availability.

Parameters:
    row (pd.Series): A row from the dataset representing a participant.
    cluster_dict (dict): A dictionary where keys are cluster IDs and values are their availabilities.

Returns:
    list: List of matching cluster IDs based on availability.
"""

    return [cluster for cluster, availability in cluster_dict.items() if any(av in availability for av in row['Availability'])]


def calculate_average_distance(element, cluster, feature_data, mentee_data):
    """
Calculates the average distance between an element and members of a specified cluster.

Parameters:
    element (pd.Series): A row from the dataset representing a participant.
    cluster (int): Cluster ID to calculate the distance to.
    feature_data (pd.DataFrame): Feature data used to compute distances.
    mentee_data (pd.DataFrame): Dataset containing mentee information, including cluster assignments.

Returns:
    float: The average distance between the element and the specified cluster.
"""

    cluster_elements = mentee_data[mentee_data['Cluster'] == cluster]
    if cluster_elements.empty:
        return np.inf
    cluster_features = feature_data.loc[cluster_elements.index]
    element_features = feature_data.loc[element.name].values.reshape(1, -1)
    distances = euclidean_distances(element_features, cluster_features)
    return distances.mean()


def reassign_cluster(row, cluster_dict, feature_data, max_cluster_size, mentee_data):
    """
Reassigns a participant to the most suitable cluster based on availability and distance constraints.

Parameters:
    row (pd.Series): A row from the dataset representing a participant.
    cluster_dict (dict): A dictionary where keys are cluster IDs and values are their availabilities.
    feature_data (pd.DataFrame): Feature data used to compute distances.
    max_cluster_size (int): Maximum allowable size for a cluster.
    mentee_data (pd.DataFrame): Dataset containing mentee information, including cluster assignments.

Returns:
    int: The ID of the reassigned cluster.
"""

    matching_clusters = find_matching_clusters(row, cluster_dict)
    if not matching_clusters:
        return row['Cluster']  # Return the original cluster if no matching clusters


    filtered_clusters = [cluster for cluster in matching_clusters if len(cluster_dict[cluster]) <= max_cluster_size]

    if filtered_clusters:
        min_distance_cluster = min(filtered_clusters,
                                   key=lambda cluster: calculate_average_distance(row, cluster, feature_data, mentee_data))
        return min_distance_cluster
    else:
        # If all matching clusters exceed the max size, choose the one with the minimum distance
        min_distance_cluster = min(matching_clusters,
                                   key=lambda cluster: calculate_average_distance(row, cluster, feature_data, mentee_data))
        return min_distance_cluster


def find_clusters_for_mentor(mentor_row, cluster_availability_dict):
    """
Finds clusters that match a mentor's availability.

Parameters:
    mentor_row (pd.Series): A row from the dataset representing a mentor.
    cluster_availability_dict (dict): Dictionary mapping clusters to their availabilities.

Returns:
    list: List of matching cluster IDs for the mentor.
"""


    return [cluster for cluster, availability in cluster_availability_dict.items() if any(av in availability for av in mentor_row['Availability'])]


def calculate_mentor_distance(mentor, cluster, feature_data, mentee_data):
    """
Calculates the average distance between a mentor and members of a specified cluster.

Parameters:
    mentor (pd.Series): A row from the dataset representing a mentor.
    cluster (int): Cluster ID to calculate the distance to.
    feature_data (pd.DataFrame): Feature data used to compute distances.
    mentee_data (pd.DataFrame): Dataset containing mentee information, including cluster assignments.

Returns:
    float: The average distance between the mentor and the specified cluster.
"""


    cluster_elements = mentee_data[mentee_data['Cluster'] == cluster]
    if cluster_elements.empty:
        return np.inf

    cluster_features = feature_data.loc[cluster_elements.index]
    mentor_features = feature_data.loc[mentor.name].values.reshape(1, -1)
    distances = euclidean_distances(mentor_features, cluster_features)

    return distances.mean()


def assign_mentor_to_cluster(mentor_row, cluster_availability_dict, feature_data, mentee_data):
    """
Assigns a mentor to the most suitable cluster based on availability and distance constraints.

Parameters:
    mentor_row (pd.Series): A row from the dataset representing a mentor.
    cluster_availability_dict (dict): Dictionary mapping clusters to their availabilities.
    feature_data (pd.DataFrame): Feature data used to compute distances.
    mentee_data (pd.DataFrame): Dataset containing mentee information, including cluster assignments.

Returns:
    int or None: The ID of the assigned cluster, or None if no suitable cluster is found.
"""

    matching_clusters = find_clusters_for_mentor(mentor_row, cluster_availability_dict)
    if not matching_clusters:
        return None
    best_cluster = min(matching_clusters,
                       key=lambda cluster: calculate_mentor_distance(mentor_row, cluster, feature_data, mentee_data))
    return best_cluster


def limit_cluster_size(data, initial_clusters, max_cluster_size, n_clusters):
    """
Adjusts initial cluster assignments to enforce size limits.

Parameters:
    data (pd.DataFrame): Dataset containing participant information.
    initial_clusters (list): Initial cluster assignments for each participant.
    max_cluster_size (int): Maximum allowable size for a cluster.
    n_clusters (int): Total number of clusters.

Returns:
    np.ndarray: Adjusted cluster assignments.
"""


    final_clusters = {i: [] for i in range(n_clusters)}


    for idx, cluster in enumerate(initial_clusters):
        if cluster < n_clusters:
            final_clusters[cluster].append(idx)


    for cluster_id in range(n_clusters):
        while len(final_clusters[cluster_id]) > max_cluster_size:


            overflow_points = final_clusters[cluster_id][max_cluster_size:]
            final_clusters[cluster_id] = final_clusters[cluster_id][:max_cluster_size]


            for point in overflow_points:

                if point >= len(data):
                    print(f"Point {point} is out of bounds for data with length {len(data)}")
                    continue


                non_current_indices = [
                    idx for c, pts in final_clusters.items() if c != cluster_id for idx in pts
                ]

                if non_current_indices:
                    nearest_cluster_indices, _ = pairwise_distances_argmin_min(
                        data.iloc[[point]], data.iloc[non_current_indices]
                    )
                    nearest_cluster_id = initial_clusters[non_current_indices[nearest_cluster_indices[0]]]


                    if len(final_clusters[nearest_cluster_id]) < max_cluster_size:
                        final_clusters[nearest_cluster_id].append(point)
                    else:

                        alternative_cluster = next(
                            (c for c in range(n_clusters) if len(final_clusters[c]) < max_cluster_size),
                            cluster_id
                        )
                        final_clusters[alternative_cluster].append(point)
                else:
                    print(f"No available clusters for point {point}")


    final_assignments = np.zeros(len(initial_clusters), dtype=int)
    for cluster_id, points in final_clusters.items():
        for point in points:
            final_assignments[point] = cluster_id

    return final_assignments


def assign_mentors_to_clusters(matching_data_mentor, cluster_availability_dict, mentor_cluster_limits):
    """
Assigns mentors to clusters, ensuring constraints such as availability and maximum cluster limits are met.

Parameters:
    matching_data_mentor (pd.DataFrame): Dataset containing mentor information.
    cluster_availability_dict (dict): Dictionary mapping clusters to their availabilities.
    mentor_cluster_limits (dict): Maximum number of clusters each mentor can be assigned to.

Returns:
    dict: Dictionary mapping clusters to their assigned mentors.
"""

    assigned_clusters = {}
    total_clusters = len(cluster_availability_dict)
    total_mentors = len(matching_data_mentor)
    max_attempts = 10000
    assigned_mentors = {} 

    for attempt in range(max_attempts):
        assigned_clusters.clear()
        assigned_mentors.clear()
        available_mentors = {cluster: [] for cluster in cluster_availability_dict}

        for idx, mentor in matching_data_mentor.iterrows():
            mentor_availability = mentor['Availability']
            for cluster, availability in cluster_availability_dict.items():
                if availability in mentor_availability:
                    available_mentors[cluster].append(idx)


        clusters = list(available_mentors.keys())
        random.shuffle(clusters)


        for cluster in clusters:
            mentors = available_mentors[cluster]
            random.shuffle(mentors)
            for mentor_idx in mentors:
                max_clusters = mentor_cluster_limits[mentor_idx] 


                if mentor_idx not in assigned_mentors:
                    assigned_mentors[mentor_idx] = []

                if len(assigned_mentors[mentor_idx]) < max_clusters:
                    if max_clusters == 2 and any(
                        cluster_availability_dict[c] == cluster_availability_dict[cluster]
                        for c in assigned_mentors[mentor_idx]
                    ):
                        continue  
                    assigned_clusters[cluster] = mentor_idx
                    assigned_mentors[mentor_idx].append(cluster)
                    break

        
        if len(assigned_clusters) == total_clusters:
            break


    for cluster, mentor_idx in assigned_clusters.items():
        mentor_availability = matching_data_mentor.loc[mentor_idx]['Availability']
        cluster_availability = cluster_availability_dict[cluster]
    for cluster in cluster_availability_dict.keys():
        if cluster not in assigned_clusters:
            print(f"Cluster {cluster} could not be assigned a mentor.")

    return assigned_clusters




def adjust_clusters( dictionary, data,min_cluster_size=12):
    """
Adjusts cluster assignments to ensure minimum cluster size and availability constraints.

Parameters:
    dictionary (dict): Dictionary mapping clusters to their availabilities.
    data (pd.DataFrame): Dataset containing cluster assignments.
    min_cluster_size (int): Minimum allowable size for a cluster.

Returns:
    pd.DataFrame: Updated dataset with adjusted cluster assignments.
"""

    cluster_sizes = data['Cluster'].value_counts().sort_values()
   
    while cluster_sizes.iloc[0] < min_cluster_size:  
   
        smallest_cluster = cluster_sizes.index[0]
        common_availability = dictionary.get(smallest_cluster, [])
       
        largest_cluster = cluster_sizes.index[-1]
     
        largest_cluster_members = data[data['Cluster'] == largest_cluster]
        candidate = largest_cluster_members[
            largest_cluster_members['Availability'].apply(lambda x: any(avail in common_availability for avail in x))
        ]
       
        if not candidate.empty:
            candidate_to_move = candidate.iloc[0]
            data.loc[data['Student_ID'] == candidate_to_move['Student_ID'], 'Cluster'] = smallest_cluster
        else:

            print(f"No candidates found for availability matching in Cluster {largest_cluster}.")
            break
       
        cluster_sizes = data['Cluster'].value_counts().sort_values()
   
    return data
##############################################
############### Main function ################
##############################################

def process_clustering_and_matching(mentee_data, mentors,mentee_info, mentor_info, n_clusters=40, max_cluster_size=18, min_cluster_size=12):
    """
    Processes mentee and mentor data to perform clustering and matching.
    
    This function clusters mentees based on hobbies, availability, and personal values, ensuring clusters meet
    size constraints. Mentors are then matched to these clusters. Finally, it outputs detailed cluster information,
    including assigned mentors and mentees.

    Parameters:
        mentee_data (DataFrame): Data about mentees (hobbies, availability, and personal values).
        mentors (DataFrame): Data about mentors (hobbies, availability, and personal values).
        mentee_info (DataFrame): Additional mentee details for concatenation.
        mentor_info (DataFrame): Additional mentor details for concatenation.
        n_clusters (int): Number of clusters to create (default is 40).
        max_cluster_size (int): Maximum allowed cluster size (default is 18).
        min_cluster_size (int): Minimum required cluster size (default is 12).

    Returns:
        cluster_info (DataFrame): Information about each cluster, including mentors and mentees.
        nouns_hobbies (DataFrame): Extracted hobbies keywords with corresponding cluster assignments.
    """
    # Map gender to numerical values for clustering purposes
    gender_mapping = {
    'Male': 1,
    'Female': 0,
    'Other/Prefer not to say': 2}
    mentee_data_original = mentee_data.copy()
    mentee_data['Gender'] = mentee_data['Gender'].map(gender_mapping)

# Extract nouns or verbs from hobbies for text-based clustering
    nouns_hobbies = [extract_nouns_or_verbs(hobby) for hobby in mentee_data['Hobbies_original']]

 # Perform KMeans clustering on hobby text features
    kmeans = Text_KMeans(n_clusters=15)
    kmeans.fit(nouns_hobbies)
    labels = kmeans.labels_
    mentee_data['Hobbies_clusters'] = labels
    matching_data = mentee_data[['Hobbies_clusters', 'Availability', 'Personal_Values']]

  # Create binary columns for each unique availability time slot
    unique_availabilities = set()
    for avail_list in matching_data['Availability']:
        unique_availabilities.update(avail_list)
    unique_availabilities = sorted(unique_availabilities)
    for avail in unique_availabilities:
        matching_data[avail] = matching_data['Availability'].apply(lambda x: 1 if avail in x else 0)
    matching_data.drop(columns=['Availability'], inplace=True)
    one_hot_encoded = pd.get_dummies(matching_data['Hobbies_clusters'], prefix='Hobby')
    one_hot_encoded2 = pd.get_dummies(matching_data['Personal_Values'])

# One-hot encode hobby clusters and personal values
    one_hot_encoded = one_hot_encoded.astype(int)
    one_hot_encoded2 = one_hot_encoded2.astype(int)
    matching_data = pd.concat([matching_data, one_hot_encoded, one_hot_encoded2], axis=1)


# Concatenate encoded features and drop original columns
    matching_data = matching_data.drop(['Hobbies_clusters', 'Personal_Values'], axis=1)

# Split data into availability and feature sets
    availability = matching_data.iloc[:, :15]
    features = matching_data.iloc[:, 15:]

# Create a combined affinity matrix for clustering
    combined_affinity_matrix = create_combined_affinity(availability, features)

    # Create a k-nearest-neighbors graph to test connectivity
    knn_graph = kneighbors_graph(features, n_neighbors=5, mode='connectivity', include_self=True)
    # Count connected components
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(knn_graph, directed=False)

    if n_components > 1:
        # The graph is disconnected - fallback to a simpler clustering for small datasets.
        if len(availability) < 50:  # If dataset is small, fallback to K-Means or fewer clusters
            n_clusters = min(n_clusters, len(availability)//2 if len(availability)//2 > 0 else 1)
            # Re-run a simpler clustering method (e.g., K-Means)
            from sklearn.cluster import KMeans
            kmeans_simplify = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans_simplify.fit_predict(features)
        else:
            # If not small dataset, just display a warning and proceed as is
            print("Warning: Graph is disconnected. Results may not be optimal.")
            # Keep existing spectral clustering code as is
            spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
            clusters = spectral_clustering.fit_predict(combined_affinity_matrix)
    else:
        # If graph is connected, proceed as usual
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        clusters = spectral_clustering.fit_predict(combined_affinity_matrix)

# Adjust cluster sizes to meet size constraints( maximum)
    clusters_size_limit = limit_cluster_size(matching_data, clusters, max_cluster_size, n_clusters)

    mentee_data['Cluster'] = clusters_size_limit
    matching_data['Cluster'] = clusters_size_limit
    mentee_data_original['Cluster'] = clusters_size_limit

 # Identify the most common availability in each cluster
    exploded_data = mentee_data.explode('Availability')
    most_common_availability = exploded_data.groupby('Cluster')['Availability'].agg(lambda x: x.mode()[0])
    most_common_availability = most_common_availability.reset_index()

# Merge common availability back into mentee data
    merged_data = pd.merge(mentee_data, most_common_availability, on='Cluster', suffixes=('', '_most_common'))

# Find mentees whose availability does not match the cluster's most common availability
    not_most_common = merged_data[~merged_data.apply(lambda row: row['Availability_most_common'] in row['Availability'], axis=1)]
    result = not_most_common[['Cluster', 'Availability']]
# Reassign mentees with mismatched availability to appropriate clusters
    cluster_availability_dict = dict(zip(most_common_availability['Cluster'], most_common_availability['Availability']))
    feature_data = matching_data.iloc[:, 15:]

    not_most_common['Reassigned_Cluster'] = not_most_common.apply(
        reassign_cluster,
        axis=1,
        cluster_dict=cluster_availability_dict,
        feature_data=feature_data,
        max_cluster_size=max_cluster_size,
        mentee_data=mentee_data_original
    )
 # Update mentee cluster assignments after reassignment
    mentee_data['Original_Cluster'] = mentee_data['Cluster'].copy()
    common_indices = mentee_data.index.intersection(not_most_common.index)
    mentee_data.loc[common_indices, 'Cluster'] = not_most_common.loc[common_indices, 'Reassigned_Cluster']
    mentee_data['Cluster'].fillna(mentee_data['Original_Cluster'], inplace=True)
    mentee_data.drop(columns='Original_Cluster', inplace=True)

    matching_data['Cluster'] = mentee_data['Cluster']
# Prepare mentor matching data
    nouns_hobbies_mentor = [extract_nouns_or_verbs(hobby) for hobby in mentors['Hobbies_original']]
    kmeans = Text_KMeans(n_clusters=15)
    kmeans.fit(nouns_hobbies_mentor)
    labels = kmeans.labels_
    mentors['Hobbies_clusters'] = labels
    matching_data_mentor = mentors[['Hobbies_clusters', 'Availability', 'Personal_Values']]

    one_hot_encoded = pd.get_dummies(matching_data_mentor['Hobbies_clusters'], prefix='Hobby')
    one_hot_encoded2 = pd.get_dummies(matching_data_mentor['Personal_Values'])
    one_hot_encoded = one_hot_encoded.astype(int)
    one_hot_encoded2 = one_hot_encoded2.astype(int)
    matching_data_mentor = pd.concat([matching_data_mentor, one_hot_encoded, one_hot_encoded2], axis=1)
    matching_data_mentor = matching_data_mentor.drop(['Hobbies_clusters', 'Personal_Values'], axis=1)


# Assign mentors to clusters based on availability
    assigned_clusters = assign_mentors_to_clusters(matching_data_mentor, cluster_availability_dict, mentors['number_of_groups'])


# Adjust mentee clusters to meet min size constraints
    mentee_data = pd.concat([ mentee_info, mentee_data], axis=1)
    mentors = pd.concat([ mentor_info, mentors], axis=1)
    mentee_data=adjust_clusters(cluster_availability_dict, mentee_data, min_cluster_size)

 # Prepare detailed cluster information
    cluster_info = []
    for cluster, group in mentee_data.groupby('Cluster'):
        cluster_number = cluster
        most_common_availability = cluster_availability_dict.get(cluster_number, 'None')
        elements = group[["First_Name", "Surname", "Student_ID", "Email", "Gender", "Domestic_International", "Hobbies_original", "Academic_Program", "Availability", "Personal_Values"]].to_dict(orient='records')
        assigned_mentor = assigned_clusters.get(cluster_number, None)
        if assigned_mentor is not None:
            mentor_details = mentor_info.loc[assigned_mentor, ["First_Name", "Surname", "Student_ID", "Email"]]
        else:
            mentor_details = {'First_Name': 'None', 'Surname': 'None', 'Student_ID': 'None', 'Email': 'None'}

        cluster_info.append({
            'Cluster': cluster_number,
            'Assigned Mentor First Name': mentor_details['First_Name'],
            'Assigned Mentor Surname': mentor_details['Surname'],
            'Assigned Mentor Student ID': mentor_details['Student_ID'],
            'Assigned Mentor Email': mentor_details['Email'],
            'Most Common Availability': most_common_availability,
            'Elements': elements,
        })

  # Create a DataFrame of hobbies keywords with cluster assignments        
        nouns_hobbies = pd.DataFrame(nouns_hobbies, columns=['Hobbies'])
        nouns_hobbies = nouns_hobbies.merge(mentee_data[['Cluster']], left_index=True, right_index=True)


    return pd.DataFrame(cluster_info), nouns_hobbies


def assign_additional_mentors_to_clusters(cluster_df, mentor_data, mentors_per_group):
    """
    Assigns additional mentors to clusters if needed.

    Parameters:
        cluster_df (pd.DataFrame): DataFrame containing cluster information with at least one mentor assigned.
        mentor_data (pd.DataFrame): DataFrame containing mentor details (must contain 'Student_ID').
        mentors_per_group (int): Number of mentors required per group, must be >= 1.

    Returns:
        pd.DataFrame: Updated cluster_df with additional mentor details if mentors_per_group > 1.
                      If no suitable additional mentors found, mentor columns remain 'None'.
    """
    if mentors_per_group <= 1:
        return cluster_df

    # Ensure mentor_data indexed by Student_ID for quick lookups
    if 'Student_ID' in mentor_data.columns:
        mentor_data = mentor_data.set_index('Student_ID', drop=False)

    # Add columns for a second mentor if not already present
    if 'Second Mentor First Name' not in cluster_df.columns:
        cluster_df['Second Mentor First Name'] = 'None'
        cluster_df['Second Mentor Surname'] = 'None'
        cluster_df['Second Mentor Student ID'] = 'None'
        cluster_df['Second Mentor Email'] = 'None'

    assigned_mentor_ids = set(cluster_df['Assigned Mentor Student ID'].unique())
    if 'None' in assigned_mentor_ids:
        assigned_mentor_ids.remove('None')

    has_availability = 'Most Common Availability' in cluster_df.columns

    # Attempt to assign additional mentors where needed
    for idx, row in cluster_df.iterrows():
        assigned_count = 1 if row['Second Mentor Student ID'] == 'None' else 2
        needed = mentors_per_group - assigned_count
        if needed > 0:
            common_avail = row['Most Common Availability'] if has_availability else None
            potential_mentors = mentor_data[~mentor_data.index.isin(assigned_mentor_ids)]
            if common_avail and 'Availability' in potential_mentors.columns:
                potential_mentors = potential_mentors[potential_mentors['Availability'].apply(lambda x: common_avail in x)]

            if not potential_mentors.empty:
                m_idx, m_row = potential_mentors.iloc[0].name, potential_mentors.iloc[0]
                cluster_df.at[idx, 'Second Mentor First Name'] = m_row.get('First_Name', 'None')
                cluster_df.at[idx, 'Second Mentor Surname'] = m_row.get('Surname', 'None')
                cluster_df.at[idx, 'Second Mentor Student ID'] = m_idx
                cluster_df.at[idx, 'Second Mentor Email'] = m_row.get('Email', 'None')
                assigned_mentor_ids.add(m_idx)

    return cluster_df

def assign_additional_mentors_from_second_file(cluster_df, second_mentor_data):
    """
    Assigns second mentors to clusters from a second mentor dataset.
    Parameters:
        cluster_df (pd.DataFrame): Clusters with at least one mentor assigned.
        second_mentor_data (pd.DataFrame): Second set of mentors for the second slot.
    Returns:
        pd.DataFrame: Updated cluster_df with second mentor assigned where possible.
    """

    # Ensure second_mentor_data indexed by Student_ID for quick lookups
    if 'Student_ID' in second_mentor_data.columns:
        second_mentor_data = second_mentor_data.set_index('Student_ID', drop=False)

    # Ensure columns for second mentor exist (they already do from previous code)
    # If not, we can create them:
    if 'Second Mentor First Name' not in cluster_df.columns:
        cluster_df['Second Mentor First Name'] = 'None'
        cluster_df['Second Mentor Surname'] = 'None'
        cluster_df['Second Mentor Student ID'] = 'None'
        cluster_df['Second Mentor Email'] = 'None'

    # Assigned mentor IDs from first file
    assigned_mentor_ids = set(cluster_df['Assigned Mentor Student ID'].unique())
    if 'None' in assigned_mentor_ids:
        assigned_mentor_ids.remove('None')

    # If we have availability in cluster_info
    has_availability = 'Most Common Availability' in cluster_df.columns

    for idx, row in cluster_df.iterrows():
        # If second mentor not assigned
        if row['Second Mentor Student ID'] == 'None':
            common_avail = row['Most Common Availability'] if has_availability else None
            potential_mentors = second_mentor_data[~second_mentor_data.index.isin(assigned_mentor_ids)]
            if common_avail and 'Availability' in potential_mentors.columns:
                potential_mentors = potential_mentors[potential_mentors['Availability'].apply(lambda x: common_avail in x)]

            if not potential_mentors.empty:
                m_idx, m_row = potential_mentors.iloc[0].name, potential_mentors.iloc[0]
                cluster_df.at[idx, 'Second Mentor First Name'] = m_row.get('First_Name', 'None')
                cluster_df.at[idx, 'Second Mentor Surname'] = m_row.get('Surname', 'None')
                cluster_df.at[idx, 'Second Mentor Student ID'] = m_idx
                cluster_df.at[idx, 'Second Mentor Email'] = m_row.get('Email', 'None')
                assigned_mentor_ids.add(m_idx)

    return cluster_df

