## Data Science and Machine Learning Portfolio

### Z1_Analysis_of_Camera-Based_Sensor_Data.ipynb
***Dataset***: Sensor observations of a city street (East Coast) on May 3rd 2020<br>
***Dataset Features***: Object unique ID, Date of observation, Time of observation, Object type, Object coordinates (square box), Object coordinates (bottom center point), Object Longitude and latitude (bottom center point)
1) Exploratory data analysis - inspect existing features
   - update object unique ID to a shortened version for easier analysis
   - combine date and time
   - split points and coordinates
2) Feature engineering
   - horizontal direction indicator for each object's overall trajectory (left-to-right or right-to-left)
   - real-time horizontal direction indicator for each object's individual observations
   - distance traveled per observation and cumulative distance traveled indicator
   - time delta per observation and cumulative time delta indicator
   - moving average for speed of object throughout observations indicator
   - current object trajectory in degrees indicator
   - real-time quadrant (40x40 pixels) information for each object
3) Data Analysis
   - What is the busiest time of the day?
   - What time during the day had the most object types (pedestrians, bicycles, trucks)?
   - How many unique objects from each object type were detected for the day?
   - How were pedestrian, bicycle and truck traffic distributed throughout the day?
   - What is the average number of observations captured by the sensor for each object type?
   - What is the average time (s), distance traveled (m) and speed (m/s) for each object type?
   - What is the most common direction of motion (left to right or right to left) for each object type? 
   - What is the average time between sensor observations for objects in each class type?
4) Visualization of Sensor Data
   - path taken by an individual object
   - paths taken by specific object type (pedestrians, bicycles or trucks) 
     - both directions 
     - left-to-right direction
     - right-to-left direction
     - irregular paths (anomalies / outliers)
   - visualize activity based on time interval
     - afternoon rush hour (11am - 1pm)
     - evening / night rush hour (5pm - 7pm)
     - midnight (11pm - 1am)
     - non-peak hours (1am - 7am)
5) Anomaly / Outlier Detection of Sensor Data
   - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
   - Quadrant count information
   - Standard Hough Line Transform (OpenCV)
   - Progressive Probabilistic Hough Line Transform (OpenCV)

### Z1_Articles_Detecting_Restricted_Content.ipynb
***Dataset***: Collection of news articles titles<br>
***Dataset Features***: Time created, Date created, Number of upvotes, Number of downvotes, Article title, Article author, Article Category, Restricted Content (Over 18) flag
1) Inspect existing features, visualization (categorical, date/time, numerical, textual)
   - punctuations in title
   - stopwords in title
   - length of title
   - number of words in title
   - average length of words in title
2) Preprocessing title column
   - remove URLs
   - remove HTML tags
   - remove emojis
   - add space around punctuations prior to removing punctuations and tokenization
   - remove punctuations
   - correct spelling of words
   - tokenization
   - remove stop words
   - stemming with PorterStemmer
   - lemmatization with WordNetLemmatizer
   - Imbalanced dataset: Undersample majority class for training / test set
   - Visualization of most common words with WordCloud
3) Feature generation and evaluate model metrics
   - Bag of Words (BoW) by manual counting
   - Bag of Words (BoW) with CountVectorizer
   - Term Frequency - Inverse Document Frequency (Tf-idf) with BoW + TfidfTransformer
   - Term Frequency - Inverse Document Frequency (Tf-idf) with TfidfVectorizer
   - Classification model evaluation (accuracy, precision, recall, confusion_matrix, roc_auc_score)
   - Analyzing titles which were false positives and false negatives
4) Word Embedding with Word2Vec
   - Continuous Bag of Words (CBOW)
   - Skip-Gram
5) Topic Modeling
   - Latent Semantic Analysis (LSA)
   - Latent Dirichlet Allocation (LDA)
   - Visualization of topic clusters with Uniform Manifold Approximation and Projection (UMAP)

### Z1_California_Housing_Prices.ipynb
***Dataset***: California housing data from the 1990 census<br>
***Dataset Features***: Longitude, Latitude, Housing median age, Total rooms, Total bedrooms, Population, Households, Median income, Ocean proximity, Median house value
1) Inspect existing features, visualization (categorical / geographic)
2) Feature engineering, data cleaning, handling text and categorical
3) Build custom transformer
   - applying imputer for missing values
   - combine existing features
   - preprocessing such as feature scaling
5) Transformation pipelines
6) Hyperparameter tuning (grid search and randomized search with cross-validation) 

### Z2_MNIST.ipynb
***Dataset***: MNIST handwritten single digits between 0 and 9<br>
***Dataset Features***: 28×28 pixel grayscale images
1) Inspect existing features, visualization
2) Exploring different models for binary and multi-class classfication
   - SGD Classifier
   - Random Forest Classifier
   - One Vs One Classifier (OVO)
3) Classification metrics 
   - Accuracy
   - Confusion matrix
   - Precision
   - Recall
   - F1-score
   - Classification report
   - Precision-Recall curve
   - ROC curve
   - ROC Area-Under-the-Curve (AUC) score
4) Preprocessing: Standard scaling inputs
5) Hyperparameter tuning (grid search and randomized search with cross-validation)
6) Feature engineering: Shifted image to increase training data
7) Using different models
   - Support Vector Machines (SVM)
   - Deep Neural Network (DNN)
   - Convolutional Neural Network (CNN)
   - Recurrent Neural Network (RNN)
   - Voting Classifier (Soft and Hard Voting)
8) Using dimensionality reduction to reduce training time
   - Principal Component Analysis (PCA)
9) Visualizing high-dimensional data
   - t-distributed Stochastic Neighbor Embedding (TSNE)

### Z2_Spam_and_Ham.ipynb
***Dataset***: Apache SpamAssassin public dataset of spam and non-spam emails<br>
***Dataset Features***: HTML and text content of email messages
1) Parsing email with email (parser, policy) Python package and string manipulation
2) Functions to get and count email structures
3) Regular expressions to remove HTML tags and convert to plain text
4) Training and test data preparation
5) Custom transformer to convert email to word counts
6) Custom transformer to convert word counts to sparse matrix
7) Transformation pipeline to combine both transformers
8) Applying classification model to training and test data
9) Classification metrics
   - Precision
   - Recall
   - Precision-Recall curve

### Z2_Titanic.ipynb
***Dataset***: Passenger information and survival status on the Titanic<br>
***Dataset Features***: Ticket class, Sex, Age in years, Number of siblings/spouses, Number of parents/children, Ticket number, Passenger fare, Cabin number, Port of embarkation, Survival status
1) Inspect existing features, visualization (numerical, categorical)
   - Pie chart
   - Bar chart
   - Crosstab
   - Correlation matrix
   - Violin plot
   - Count plot
   - Facet grid
   - Factor plot
2) Feature engineering
3) Transformation pipeline for numerical and categorical data
4) Exploring different models for binary classification
   - KNeighborsClassifier
   - Support Vector Classifier (SVC)
   - Gaussian Naive Bayes
   - Decision Tree Classifier
   - Random Forest Classifier
   - Extra Trees Classifier
   - AdaBoost Classifier
   - Gradient Boosting Classifier
   - Linear Discriminant Analysis (LDA)
   - Voting Classifier
5) Check for sensitivity to data with cross-validation
6) Hyperparameter tuning (grid search and randomized search with cross-validation)
7) Compare feature importances for different models

### Z9_Olivetti.ipynb
***Dataset***: A set of face images taken between April 1992 and April 1994 at AT&T Laboratories Cambridge<br>
***Dataset Features***: 64×64 pixel grayscale images
1) Inspect existing features, visualization (numerical)
2) Stratify training, test and validation set due to unbalanced dataset
3) Using dimensionality reduction to reduce feature size and training time
4) Clustering for unlabelled data (unsupervised learning)
   - KMeans clustering
5) Evaluating number of clusters for KMeans model
   - Elbow method based on inertia
   - Silhouette score 
6) Visualizing clusters to check for similarity
7) Verifying classifier accuracy
   - Logistic Regression
   - Random Forest Classifier
   - SGD Classifier
8) Using KMeans for dimensionality reduction
9) Model prediction pipeline
10) Hyperparameter tuning (grid search with cross-validation)
11) Classification with Gaussian Mixture Model (GMM) 
12) Anomaly detection with Gaussian Mixture Model
