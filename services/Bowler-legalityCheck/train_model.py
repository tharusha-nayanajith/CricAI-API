from features.bowlingActionsChecker.bowlingactions import process_professional_images, build_dataset_from_features, train_and_save_model

# Step 1: Extract features from professional images (legal/illegal)
print("Extracting features from professional images...")
process_professional_images()

# Step 2: Build dataset (X = features, y = labels)
print("Building dataset...")
X, y = build_dataset_from_features()

# Step 3: Train and save model + scaler
print("Training model...")
train_and_save_model(X, y)

print("Training completed. Model saved in 'models/bowler_model.h5'")
