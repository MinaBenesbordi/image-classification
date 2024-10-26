# Image Classification with Graph-Based Feature Extraction

This project implements a classification model that utilizes image processing and machine learning techniques to classify images based on features extracted from superpixel segmentation and keypoint detection. The code reads images from a specified directory, processes them to create feature graphs, and then applies various classifiers to predict categories.

## Features

- **Image Processing**: 
  - Reads images from a specified folder.
  - Applies superpixel segmentation to images.
  - Extracts keypoints using the SIFT algorithm.
  
- **Graph Construction**: 
  - Creates graphs based on spatial relationships and similarities between nodes and keypoints.
  
- **Feature Extraction**: 
  - Computes centrality measures and clustering coefficients from the constructed graphs.

- **Classification**:
  - Implements multiple classifiers, including Decision Tree, Random Forest, K-Nearest Neighbors, and Voting Classifier, among others.
  - Evaluates the accuracy of each classifier on the test set.

## Requirements

- Python 3.x
- OpenCV
- scikit-learn
- NumPy
- pandas
- networkx
- scikit-image

You can install the required libraries using pip:

```bash
pip install opencv-python scikit-learn numpy pandas networkx scikit-image
```

## Dataset

Place your image dataset in a folder. Each image file should be named with a number indicating its category, for example, `100.jpg`, `200.png`, etc. Ensure the path to the dataset folder is correctly specified in the `main()` function.

## Code Structure

### 1. Image Reading

The `readImages` function reads images from a specified directory and extracts their category from the filename.

### 2. Superpixel Segmentation

The `superpixelSegmentation` function applies superpixel segmentation to an image using the SLIC algorithm.

### 3. Keypoint Extraction

The `extract_keypoints` function uses the SIFT algorithm to detect keypoints in the image.

### 4. Similarity Computation

The `similarity_nodes` and `similarity_keypoints` functions calculate similarity metrics between nodes and keypoints.

### 5. Graph Creation

The `create_graph` function constructs graphs based on specified similarity thresholds and Euclidean distances between nodes and keypoints.

### 6. Feature Extraction

The `extract_features` function computes various centrality measures and clustering coefficients from the graphs.

### 7. Classification

The `main` function orchestrates the workflow:
- Reads images and processes them to create graphs and extract features.
- Writes features to CSV files.
- Reads the CSV files and applies classifiers to predict the outcomes.
- Evaluates the accuracy of each classifier and prints the results.

## Example Usage

To run the project, make sure to set the correct dataset path in the `main()` function:

```python
dataset_folder = 'path/to/your/image/folder'
```

Then execute the script:

```bash
python script.py
```

## Output

The output will display the accuracy of each classifier on the test set. It will also print a sorted DataFrame showing each algorithm and its corresponding accuracy.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
