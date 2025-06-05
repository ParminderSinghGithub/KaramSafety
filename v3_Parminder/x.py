# class DataProcessor:
#     def __init__(self, batch_size: int = 31, random_state: int = 43):
#         self.batch_size = batch_size
#         self.random_state = random_state
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
        
#     def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
#         cols_to_drop = ['Unnamed: 0', 'time']
#         X_clean = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
#         X_clean = X_clean.dropna()
#         return X_clean
    
#     def create_batches(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         n_samples = (len(X) // self.batch_size) * self.batch_size
#         X_trimmed = X[:n_samples]
#         y_trimmed = y[:n_samples]
        
#         X_batched = X_trimmed.reshape(-1, self.batch_size, X_trimmed.shape[1])
#         y_batched = y_trimmed.reshape(-1, self.batch_size)
        
#         X_aggregated = np.mean(X_batched, axis=1)
#         y_aggregated = np.array([np.bincount(batch).argmax() for batch in y_batched])
        
#         return X_aggregated, y_aggregated
    
#     def fit_transform_train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         X_clean = self.clean_data(X_train)
#         valid_indices = X_clean.index
#         X_values = X_clean.values
#         y_aligned = y_train[valid_indices]
        
#         shuffled_indices = shuffle(range(len(X_values)), random_state=self.random_state)
#         X_shuffled = X_values[shuffled_indices]
#         y_shuffled = y_aligned[shuffled_indices]
        
#         y_encoded = self.label_encoder.fit_transform(y_shuffled)
#         X_batched, y_batched = self.create_batches(X_shuffled, y_encoded)
#         X_scaled = self.scaler.fit_transform(X_batched)
#         y_final = self.label_encoder.inverse_transform(y_batched)
        
#         return X_scaled, y_final
    
#     def transform_test(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         X_clean = self.clean_data(X_test)
#         valid_indices = X_clean.index
#         X_values = X_clean.values
#         y_aligned = y_test[valid_indices]
#         X_scaled = self.scaler.transform(X_values)
        
#         return X_scaled, y_aligned

# print(f"Processed Training Batches: {len(X_train_processed):,}")
# print(f"Batch Size: 31")


# # Random Forest and Decision Tree Models
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Random Forest Model
# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     criterion='gini',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features='sqrt',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=-1,
#     random_state=43,
#     verbose=0,
#     warm_start=False,
#     class_weight=None,
#     ccp_alpha=0.0,
#     max_samples=None
# )

# rf_model.fit(X_train_processed, y_train_processed)

# rf_y_pred = rf_model.predict(X_test_processed)
# rf_accuracy = accuracy_score(y_test_processed, rf_y_pred)
# rf_cm = confusion_matrix(y_test_processed, rf_y_pred)
# rf_classification_rep = classification_report(y_test_processed, rf_y_pred, output_dict=True)

# print("Random Forest Model Performance Evaluation\n")
# print(f"Test Accuracy: {rf_accuracy:.6f}")
# print(f"Test Samples: {len(y_test_processed):,}")

# rf_results_df = pd.DataFrame(rf_classification_rep).transpose()
# print("\nClassification Report:")
# print(rf_results_df.round(4))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Confusion Matrix
# sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=rf_model.classes_,
#             yticklabels=rf_model.classes_, ax=ax1)
# ax1.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Predicted')
# ax1.set_ylabel('Actual')

# # Per-Class Performance Metrics
# metrics = ['precision', 'recall', 'f1-score']
# classes = [c for c in rf_results_df.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
# x_pos = np.arange(len(classes))

# for i, metric in enumerate(metrics):
#     values = [rf_results_df.loc[c, metric] for c in classes]
#     ax2.bar(x_pos + i*0.25, values, width=0.25, label=metric, alpha=0.8)

# ax2.set_xlabel('Classes')
# ax2.set_ylabel('Score')
# ax2.set_title('Random Forest - Per-Class Performance Metrics', fontsize=14, fontweight='bold')
# ax2.set_xticks(x_pos + 0.25)
# ax2.set_xticklabels(classes, rotation=45)
# ax2.legend()
# ax2.set_ylim(0, 1)

# plt.tight_layout()
# plt.show()

# # Decision Tree Model
# dt_model = DecisionTreeClassifier(
#     criterion='gini',
#     splitter='best',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features=None,
#     random_state=43,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     class_weight=None,
#     ccp_alpha=0.0
# )

# dt_model.fit(X_train_processed, y_train_processed)

# dt_y_pred = dt_model.predict(X_test_processed)
# dt_accuracy = accuracy_score(y_test_processed, dt_y_pred)
# dt_cm = confusion_matrix(y_test_processed, dt_y_pred)
# dt_classification_rep = classification_report(y_test_processed, dt_y_pred, output_dict=True)

# print("\n\nDecision Tree Model Performance Evaluation\n")
# print(f"Test Accuracy: {dt_accuracy:.6f}")
# print(f"Test Samples: {len(y_test_processed):,}")

# dt_results_df = pd.DataFrame(dt_classification_rep).transpose()
# print("\nClassification Report:")
# print(dt_results_df.round(4))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Confusion Matrix
# sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=dt_model.classes_,
#             yticklabels=dt_model.classes_, ax=ax1)
# ax1.set_title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Predicted')
# ax1.set_ylabel('Actual')

# # Per-Class Performance Metrics
# for i, metric in enumerate(metrics):
#     values = [dt_results_df.loc[c, metric] for c in classes]
#     ax2.bar(x_pos + i*0.25, values, width=0.25, label=metric, alpha=0.8)

# ax2.set_xlabel('Classes')
# ax2.set_ylabel('Score')
# ax2.set_title('Decision Tree - Per-Class Performance Metrics', fontsize=14, fontweight='bold')
# ax2.set_xticks(x_pos + 0.25)
# ax2.set_xticklabels(classes, rotation=45)
# ax2.legend()
# ax2.set_ylim(0, 1)

# plt.tight_layout()
# plt.show()

# # Save Models
# joblib.dump(rf_model, "random_forest_model.pkl")
# joblib.dump(dt_model, "decision_tree_model.pkl")

# print("\n\nModel artifacts saved:")
# print("   ✓ random_forest_model.pkl")
# print("   ✓ decision_tree_model.pkl")

# # Model sizes
# rf_model_size_mb = os.path.getsize("random_forest_model.pkl") / (1024 * 1024)
# dt_model_size_kb = os.path.getsize("decision_tree_model.pkl") / 1024
# print(f"\nRandom Forest Model Size: {rf_model_size_mb:.2f} MB")
# print(f"Decision Tree Model Size: {dt_model_size_kb:.2f} KB")