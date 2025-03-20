from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Read the data from the text file
with open('output.txt', 'r') as f:
    data = f.readlines()

# Step 2: Extract actual and predicted labels
actual = []
predicted = []

for line in data:
    actual_label, predicted_label = line.strip().split(',')
    actual.append(actual_label)
    predicted.append(predicted_label)

# Step 3: Calculate metrics

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)

# Calculate precision (average='macro' for multiple classes)
precision = precision_score(actual, predicted, average='macro', zero_division=0)

# Calculate recall (average='macro' for multiple classes)
recall = recall_score(actual, predicted, average='macro', zero_division=0)

# Calculate F1 score (average='macro' for multiple classes)
f1 = f1_score(actual, predicted, average='macro', zero_division=0)

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')