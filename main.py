import os
import csv
import ast
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def readImages(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Extract the left number from the image filename as the category
                fileNum = int(filename.split(".")[0])
                category = fileNum//100
                images.append((image, category))
    return images

def superpixelSegmentation(image):
    segments = slic(image, n_segments=250, compactness=10, sigma=1, start_label=1)
    labels = segments.reshape(-1)

    regions = regionprops(segments)
    nodes = [(props.centroid[1], props.centroid[0]) for props in regions]

    return nodes

def extract_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(image, None)
    return keypoints

def similarity_nodes(node1, node2):
    x1, y1 = node1
    x2, y2 = node2

    intensity_diff = np.abs(x1 - x2) + np.abs(y1 - y2)
    similarity = np.mean(intensity_diff)

    return similarity

def similarity_keypoints(keypoint1, keypoint2, image):
    x1, y1 = map(int, keypoint1.pt)
    x2, y2 = map(int, keypoint2.pt)

    intensity1 = image[y1, x1]
    intensity2 = image[y2, x2]

    similarity = np.mean(np.abs(intensity1 - intensity2))

    return similarity

def create_graph(nodes, keypoints, threshold1, threshold2, image):
    graph1 = nx.Graph()
    graph2 = nx.Graph()

    num_nodes = len(nodes)
    for i in range(num_nodes):
        node1 = nodes[i]
        x1, y1 = node1

        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            x2, y2 = node2

            euclidean_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            similarity = similarity_nodes(node1, node2)

            if euclidean_distance < threshold1:
                graph1.add_edge(i, j)
            if similarity > threshold2:
                graph2.add_edge(i, j)

    graph3 = nx.Graph()
    graph4 = nx.Graph()
    num_keypoints = len(keypoints)
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            keypoint1 = keypoints[i]
            keypoint2 = keypoints[j]
            x1, y1 = map(int, keypoint1.pt)
            x2, y2 = map(int, keypoint2.pt)

            euclidean_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            similarity = similarity_keypoints(keypoint1, keypoint2, image)

            if euclidean_distance < threshold1:
                graph3.add_edge(i, j)
            if similarity > threshold2:
                graph4.add_edge(i, j)

    return graph1, graph2, graph3, graph4

def extract_features(graph):
    feature_vectors = []
    
    bet_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
    degree_centrality = nx.degree_centrality(graph)
    clustering_coefficient = nx.clustering(graph)
    closeness_centrality = nx.closeness_centrality(graph)
        
    feature_vector = [list(bet_centrality.values()),
                     list(degree_centrality.values()),
                     list(clustering_coefficient.values()),
                     list(closeness_centrality.values())]
        
    feature_vectors.append(feature_vector)

    return feature_vectors

def main():
    dataset_folder = 'D:/Uni/04/project/40015092001-40015091021/0000'
    T1 = 20  # Threshold 1 for Euclidean distance
    T2 = 0.9  # Threshold 2 for similarity

    images = readImages(dataset_folder)

    for idx, image in enumerate(images):
        # Extract category from the images list
        category = image[1]
        nodes = superpixelSegmentation(image[0])
        keypoints = extract_keypoints(image[0])
        graph1, graph2, graph3, graph4 = create_graph(nodes, keypoints, T1, T2, image[0])
        graphs = [(graph1, 'graph1'), (graph2, 'graph2'), (graph3, 'graph3'), (graph4, 'graph4')]
        for graph, graph_name in graphs:
            feature_vectors = extract_features(graph)
            output_filename = f'{graph_name}.csv'
            features = [[category] + feature_vector for feature_vector in feature_vectors]
            with open(output_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(features)


    csv_files = [['graph1.csv'], ['graph2.csv'], ['graph3.csv'], ['graph4.csv']]
    result = []

    for csv_file_list in csv_files:
        result.clear()  # Clear the result list for each iteration
        df = pd.read_csv(csv_file_list[0])
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        X = imputer.fit_transform(X)

        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        encoder = OrdinalEncoder()
        X = encoder.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classifiers = [
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            KNeighborsClassifier(n_neighbors=7),
            BaggingClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            SVC(),
            LogisticRegression(max_iter=5000),
            SGDClassifier(),
            VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB())])
        ]

        for classifier in classifiers:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            result.append(accuracy)

        print(result)
        output = pd.DataFrame({"Algorithm": ["Decision Tree", "Random Forest", "K-Nearest Neighbor", "Bagging", "Boosting", "Naive Bayes", "SVM", "Logistic Regression", "SGD", "Voting"], "Accuracy": result})
        output = output.sort_values(by="Accuracy", ascending=False)
        print(output)

if __name__ == '__main__':
    main()
