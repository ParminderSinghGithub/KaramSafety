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




# class DataLoader:
#     def __init__(self):
#         self.train_data = None
#         self.test_data = None
        
#     def load_data(self, root_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
#         data_frames = []
        
#         for folder in sorted(os.listdir(root_dir)):
#             folder_path = os.path.join(root_dir, folder)
#             if not os.path.isdir(folder_path):
#                 continue
                
#             for file in os.listdir(folder_path):
#                 if not file.endswith('.csv'):
#                     continue
                    
#                 file_path = os.path.join(folder_path, file)
#                 df = pd.read_csv(file_path, encoding="ISO-8859-1")
#                 df['label'] = folder
#                 data_frames.append(df)
        
#         combined = pd.concat(data_frames, ignore_index=True)
#         X = combined.drop(columns=['label'])
#         y = combined['label'].values
        
#         return X, y

# loader = DataLoader()
# X_train_raw, y_train_raw = loader.load_data('train')
# X_test_raw, y_test_raw = loader.load_data('test')

# print(f"Training Data Shape: {X_train_raw.shape}")
# print(f"Test Data Shape: {X_test_raw.shape}")
# print(f"Features: {X_train_raw.shape[1]}")
# print(f"Classes: {len(np.unique(y_train_raw))}")




# class DataProcessor:
#     def __init__(self, random_state: int = 43):
#         self.random_state = random_state
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
        
#     def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
#         cols_to_drop = ['Unnamed: 0', 'time']
#         X_clean = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
#         X_clean = X_clean.dropna()
#         return X_clean
    
#     def fit_transform_train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         X_clean = self.clean_data(X_train)
#         valid_indices = X_clean.index
#         X_values = X_clean.values
#         y_aligned = y_train[valid_indices]
        
#         shuffled_indices = shuffle(range(len(X_values)), random_state=self.random_state)
#         X_shuffled = X_values[shuffled_indices]
#         y_shuffled = y_aligned[shuffled_indices]
        
#         y_encoded = self.label_encoder.fit_transform(y_shuffled)
#         X_scaled = self.scaler.fit_transform(X_shuffled)
#         y_final = self.label_encoder.inverse_transform(y_encoded)
        
#         return X_scaled, y_final
    
#     def transform_test(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         X_clean = self.clean_data(X_test)
#         valid_indices = X_clean.index
#         X_values = X_clean.values
#         y_aligned = y_test[valid_indices]
#         X_scaled = self.scaler.transform(X_values)
        
#         return X_scaled, y_aligned
        
# processor = DataProcessor()
# X_train_processed, y_train_processed = processor.fit_transform_train(X_train_raw, y_train_raw)
# X_test_processed, y_test_processed = processor.transform_test(X_test_raw, y_test_raw)

# print("Data Processing Summary:")
# print(f"Original Training Samples: {len(X_train_raw):,}")
# print(f"Processed Training Samples: {len(X_train_processed):,}")
# print(f"Test Samples: {len(X_test_processed):,}")









# class DataLoader:
#     def __init__(self, train_split=0.7, random_state=43):
#         self.train_split = train_split
#         self.random_state = random_state
#         self.all_data = None
#         self.all_labels = None
#         self.worker_metadata = None
#         self.train_workers = None
#         self.test_workers = None

#     def load_all_data(self, train_dir: str, test_dir: str) -> None:

#         all_data_frames = []
#         all_metadata = []

#         for directory, dir_type in [(train_dir, 'original_train'), (test_dir, 'original_test')]:
#             if not os.path.exists(directory):
#                 continue

#             for folder in sorted(os.listdir(directory)):
#                 folder_path = os.path.join(directory, folder)
#                 if not os.path.isdir(folder_path):
#                     continue

#                 for file in os.listdir(folder_path):
#                     if not file.endswith('.csv'):
#                         continue

#                     file_path = os.path.join(folder_path, file)
#                     df = pd.read_csv(file_path, encoding="ISO-8859-1")
#                     worker_id = f"{dir_type}_{folder}_{file.replace('.csv', '')}"

#                     for idx in range(len(df)):
#                         all_metadata.append({
#                             'worker_id': worker_id,
#                             'activity': folder,
#                             'original_source': dir_type,
#                             'file_path': file_path,
#                             'row_in_file': idx
#                         })

#                     df['label'] = folder
#                     all_data_frames.append(df)

#         self.all_data = pd.concat(all_data_frames, ignore_index=True)
#         self.worker_metadata = pd.DataFrame(all_metadata)
#         self.all_labels = self.all_data['label'].values

#         print(f"Total samples: {len(self.all_data)}")
#         print(f"Unique workers: {self.worker_metadata['worker_id'].nunique()}")
#         print(f"Activities: {sorted(self.worker_metadata['activity'].unique())}")

#         self._analyze_worker_distribution()

#     def _analyze_worker_distribution(self):
#         print("Worker distribution per activity:")
#         for activity in sorted(self.worker_metadata['activity'].unique()):
#             workers = self.worker_metadata[self.worker_metadata['activity'] == activity]['worker_id'].nunique()
#             samples = len(self.worker_metadata[self.worker_metadata['activity'] == activity])
#             print(f"{activity}: {workers} workers, {samples} samples")
#         print(f"Total: {self.worker_metadata['worker_id'].nunique()} workers, {len(self.worker_metadata)} samples")

#     def split_by_workers(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
#         if self.all_data is None:
#             raise ValueError("Data not loaded.")

#         all_workers = self.worker_metadata['worker_id'].unique()
#         np.random.seed(self.random_state)
#         np.random.shuffle(all_workers)

#         n_train_workers = int(len(all_workers) * self.train_split)
#         self.train_workers = all_workers[:n_train_workers]
#         self.test_workers = all_workers[n_train_workers:]

#         print(f"Train workers: {len(self.train_workers)}")
#         print(f"Test workers: {len(self.test_workers)}")

#         train_mask = self.worker_metadata['worker_id'].isin(self.train_workers)
#         test_mask = self.worker_metadata['worker_id'].isin(self.test_workers)

#         X_train = self.all_data[train_mask].drop(columns=['label']).reset_index(drop=True)
#         X_test = self.all_data[test_mask].drop(columns=['label']).reset_index(drop=True)
#         y_train = self.all_labels[train_mask]
#         y_test = self.all_labels[test_mask]

#         print(f"Train samples: {len(X_train)}")
#         print(f"Test samples: {len(X_test)}")

#         self._check_activity_distribution(train_mask, test_mask)

#         return X_train, X_test, y_train, y_test

#     def _check_activity_distribution(self, train_mask, test_mask):
#         train_activities = set(self.worker_metadata[train_mask]['activity'].unique())
#         test_activities = set(self.worker_metadata[test_mask]['activity'].unique())
#         all_activities = set(self.worker_metadata['activity'].unique())

#         print(f"Train activities: {sorted(train_activities)}")
#         print(f"Test activities: {sorted(test_activities)}")

#         if all_activities - train_activities:
#             print(f"Missing in train: {sorted(all_activities - train_activities)}")
#         if all_activities - test_activities:
#             print(f"Missing in test: {sorted(all_activities - test_activities)}")

#     def load_data(self, root_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
#         if root_dir.lower() == 'train':
#             if self.all_data is None:
#                 raise ValueError("Call load_all_data first.")
#             train_mask = self.worker_metadata['worker_id'].isin(self.train_workers)
#             X = self.all_data[train_mask].drop(columns=['label']).reset_index(drop=True)
#             y = self.all_labels[train_mask]
#             return X, y
#         elif root_dir.lower() == 'test':
#             if self.all_data is None:
#                 raise ValueError("Call load_all_data first.")
#             test_mask = self.worker_metadata['worker_id'].isin(self.test_workers)
#             X = self.all_data[test_mask].drop(columns=['label']).reset_index(drop=True)
#             y = self.all_labels[test_mask]
#             return X, y
#         else:
#             raise ValueError("Invalid root_dir. Use 'train' or 'test'.")

#     def get_train_test_split(self, train_dir: str = 'train', test_dir: str = 'test') -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
#         self.load_all_data(train_dir, test_dir)
#         return self.split_by_workers()


# class DataProcessor:
#     def __init__(self, random_state: int = 43):
#         self.random_state = random_state
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()

#     def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
#         cols_to_drop = ['Unnamed: 0', 'time']
#         X_clean = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
#         return X_clean.dropna()

#     def fit_transform_train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

#         X_clean = self.clean_data(X_train)
#         print(f"X_clean shape: {X_clean.shape}")

#         valid_indices = X_clean.index.tolist()
#         y_aligned = y_train if len(valid_indices) == len(y_train) else y_train[valid_indices]

#         X_values = X_clean.values
#         shuffled_indices = shuffle(range(len(X_values)), random_state=self.random_state)
#         X_shuffled = X_values[shuffled_indices]
#         y_shuffled = y_aligned[shuffled_indices]

#         y_encoded = self.label_encoder.fit_transform(y_shuffled)
#         X_scaled = self.scaler.fit_transform(X_shuffled)
#         y_final = self.label_encoder.inverse_transform(y_encoded)

#         print(f"X_scaled shape: {X_scaled.shape}, y_final shape: {y_final.shape}")
#         return X_scaled, y_final

#     def transform_test(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#         X_clean = self.clean_data(X_test)
#         print(f"X_clean shape: {X_clean.shape}")

#         valid_indices = X_clean.index.tolist()
#         y_aligned = y_test if len(valid_indices) == len(y_test) else y_test[valid_indices]

#         X_values = X_clean.values
#         X_scaled = self.scaler.transform(X_values)

#         print(f"X_scaled shape: {X_scaled.shape}, y_aligned shape: {y_aligned.shape}")
#         return X_scaled, y_aligned

# loader = DataLoader(train_split=0.8, random_state=43)
# X_train_raw, X_test_raw, y_train_raw, y_test_raw = loader.get_train_test_split('train', 'test')

# print(f"X_train shape: {X_train_raw.shape}")
# print(f"X_test shape: {X_test_raw.shape}")
# print(f"Features: {X_train_raw.shape[1]}")
# print(f"Classes: {len(np.unique(y_train_raw))}")

# processor = DataProcessor()
# X_train_processed, y_train_processed = processor.fit_transform_train(X_train_raw, y_train_raw)
# X_test_processed, y_test_processed = processor.transform_test(X_test_raw, y_test_raw)

# print(f"Processed X_train: {X_train_processed.shape}")
# print(f"Processed X_test: {X_test_processed.shape}")









# model = LogisticRegression(
#     penalty='l2',
#     dual=False,
#     tol=1e-3,
#     C=0.089,
#     fit_intercept=True,
#     intercept_scaling=0.80,
#     class_weight=None,
#     solver='liblinear',
#     max_iter=100,
#     multi_class='ovr',
#     verbose=0,
#     warm_start=False,
#     n_jobs=-1,
#     random_state=43
# )

# model.fit(X_train_processed, y_train_processed)

# y_pred = model.predict(X_test_processed)
# accuracy = accuracy_score(y_test_processed, y_pred)
# cm = confusion_matrix(y_test_processed, y_pred)
# classification_rep = classification_report(y_test_processed, y_pred, output_dict=True)

# print("Model Performance Evaluation\n")
# print(f"Test Accuracy: {accuracy:.6f}")
# print(f"Test Samples: {len(y_test_processed):,}")

# results_df = pd.DataFrame(classification_rep).transpose()
# print("\nClassification Report:")
# print(results_df.round(4))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Confusion Matrix
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=model.classes_, 
#             yticklabels=model.classes_, ax=ax1)
# ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Predicted')
# ax1.set_ylabel('Actual')

# # Per-Class Performance Metrics
# metrics = ['precision', 'recall', 'f1-score']
# classes = [c for c in results_df.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
# x_pos = np.arange(len(classes))

# for i, metric in enumerate(metrics):
#     values = [results_df.loc[c, metric] for c in classes]
#     ax2.bar(x_pos + i*0.25, values, width=0.25, label=metric, alpha=0.8)

# ax2.set_xlabel('Classes')
# ax2.set_ylabel('Score')
# ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
# ax2.set_xticks(x_pos + 0.25)
# ax2.set_xticklabels(classes, rotation=45)
# ax2.legend()
# ax2.set_ylim(0, 1)

# plt.tight_layout()
# plt.show()

# joblib.dump(model, "logistic_regression_model.pkl")
# joblib.dump(processor.scaler, "scaler.pkl")
# joblib.dump(processor.label_encoder, "label_encoder.pkl")

# model_size_kb = os.path.getsize("logistic_regression_model.pkl") / 1024
# print(f"\nModel Size: {model_size_kb:.2f} KB")
