import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv(r'C:\Users\nikhi\Desktop\Image_Imentiv\emotion\omg_test_dataset\current_imentiv_model\emotion_results_current_imentiv_model.csv')

# Get the true labels and predicted labels
true_labels = df['label']
predicted_labels = df['dominant_emotion']

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Calculate the accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the accuracy
print("Accuracy:", accuracy)

# Get the unique labels from the confusion matrix
labels = np.unique(np.concatenate((true_labels, predicted_labels)))

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()