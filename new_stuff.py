
from collections import defaultdict


def compare_clustering_results(task1_features, task2_features):
    # Create dictionaries to store cluster assignments for each task
    task1_clusters = defaultdict(set)
    task2_clusters = defaultdict(set)

    # Populate the dictionaries with feature-cluster assignments
    for feature, cluster in task1_features:
        task1_clusters[cluster].add(feature)
    for feature, cluster in task2_features:
        task2_clusters[cluster].add(feature)

    # Find the common features between the two tasks
    common_features = set(task1_clusters.keys()) & set(task2_clusters.keys())

    # Calculate the total number of features in each task
    task1_total_features = sum(len(features) for features in task1_clusters.values())
    task2_total_features = sum(len(features) for features in task2_clusters.values())

    # Calculate the percentage of common features
    common_features_percentage = len(common_features) / (task1_total_features + task2_total_features) * 100

    # Calculate the percentage of features assigned to the same cluster in both tasks
    same_cluster_count = 0
    for feature in common_features:
        task1_cluster = next(cluster for cluster, features in task1_clusters.items() if feature in features)
        task2_cluster = next(cluster for cluster, features in task2_clusters.items() if feature in features)
        if task1_cluster == task2_cluster:
            same_cluster_count += 1
    same_cluster_percentage = same_cluster_count / len(common_features) * 100

    # Calculate the average number of features per cluster in each task
    task1_avg_features_per_cluster = task1_total_features / len(task1_clusters)
    task2_avg_features_per_cluster = task2_total_features / len(task2_clusters)

    # Return the statistics as a dictionary
    stats = {
        "common_features_percentage": common_features_percentage,
        "same_cluster_percentage": same_cluster_percentage,
        "task1_avg_features_per_cluster": task1_avg_features_per_cluster,
        "task2_avg_features_per_cluster": task2_avg_features_per_cluster
    }
    return stats


    compare_clustering_results(task1_features, task2_features)