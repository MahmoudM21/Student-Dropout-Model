import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score, accuracy_score, 
                           f1_score, classification_report, confusion_matrix, 
                           roc_auc_score, mean_absolute_error)
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentDropoutPredictor:
    def __init__(self, data_path='./dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Handle duplicates
            duplicates = self.df.duplicated().sum()
            if duplicates > 0:
                self.df.drop_duplicates(inplace=True)
                print(f"Removed {duplicates} duplicate rows")
            
            # Handle missing values
            missing_values = self.df.isnull().sum().sum()
            if missing_values > 0:
                print(f"Found {missing_values} missing values")
                # Fill missing values with median for numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # Encode target variable
            if 'Target' in self.df.columns:
                label_encoder = LabelEncoder()
                self.df['Target'] = label_encoder.fit_transform(self.df['Target'])
                print("Target classes:", self.df['Target'].value_counts().to_dict())
                
                # Remove enrolled students (assuming 1 is enrolled)
                if 1 in self.df['Target'].unique():
                    self.df = self.df[self.df['Target'] != 1].copy()
                    print("Removed enrolled students")
                
                # Create binary dropout column (0: Graduate, 1: Dropout)
                self.df['Dropout'] = self.df['Target'].apply(lambda x: 1 if x == 0 else 0)
                print("Dropout distribution:", self.df['Dropout'].value_counts().to_dict())
            
            # Store feature names (first 34 columns)
            self.feature_names = self.df.columns[:34].tolist()
            print(f"Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def prepare_data(self):
        """Prepare data for training"""
        try:
            # Features and target
            X = self.df.iloc[:, :34].values
            y = self.df['Dropout'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=1, stratify=y
            )
            
            print(f"Training set: {self.x_train.shape}, Test set: {self.x_test.shape}")
            return True
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return False
    
    def train_models(self):
        """Train multiple models"""
        try:
            # Initialize models
            models_config = {
                'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000),
                'SVM': SVC(random_state=1, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=1)
            }
            
            # Train models
            for name, model in models_config.items():
                print(f"Training {name}...")
                model.fit(self.x_train, self.y_train)
                self.models[name] = model
                
                # Evaluate model
                y_pred = model.predict(self.x_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f"{name} Accuracy: {accuracy:.4f}")
            
            print("All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return False
    
    def evaluate_model(self, model_name):
        """Evaluate a specific model"""
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        y_pred = model.predict(self.x_test)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, y_pred)
        }
        
        return metrics
    
    def predict_dropout(self, input_data, model_name='Decision Tree'):
        """Make prediction for new data"""
        try:
            if model_name not in self.models:
                return None, "Model not found"
            
            # Convert input to numpy array
            if isinstance(input_data, list):
                input_array = np.array(input_data).reshape(1, -1)
            else:
                input_array = input_data.reshape(1, -1)
            
            # Scale input data
            input_scaled = self.scaler.transform(input_array)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Get probability for dropout (class 1)
            dropout_probability = probability[1] if len(probability) > 1 else probability[0]
            
            return prediction, dropout_probability
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"
    
    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_names if self.feature_names else []
    
    def get_model_names(self):
        """Get list of available model names"""
        return list(self.models.keys())
    
    def save_models(self, filepath_prefix='dropout_model'):
        """Save trained models and scaler"""
        try:
            # Save scaler
            joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
            
            # Save models
            for name, model in self.models.items():
                safe_name = name.replace(' ', '_').lower()
                joblib.dump(model, f'{filepath_prefix}_{safe_name}.pkl')
            
            # Save feature names
            joblib.dump(self.feature_names, f'{filepath_prefix}_features.pkl')
            
            print("Models saved successfully!")
            return True
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, filepath_prefix='dropout_model'):
        """Load trained models and scaler"""
        try:
            # Load scaler
            self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
            
            # Load feature names
            self.feature_names = joblib.load(f'{filepath_prefix}_features.pkl')
            
            # Load models
            model_files = {
                'Logistic Regression': f'{filepath_prefix}_logistic_regression.pkl',
                'SVM': f'{filepath_prefix}_svm.pkl',
                'Decision Tree': f'{filepath_prefix}_decision_tree.pkl'
            }
            
            for name, filepath in model_files.items():
                try:
                    self.models[name] = joblib.load(filepath)
                except:
                    print(f"Could not load {name} model")
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

# Function to initialize and train the model
def setup_model(data_path='dataset.csv'):
    """Setup and train the dropout prediction model"""
    predictor = StudentDropoutPredictor(data_path)
    
    if predictor.load_and_preprocess_data():
        if predictor.prepare_data():
            if predictor.train_models():
                predictor.save_models()
                return predictor
    
    return None

if __name__ == "__main__":
    # Example usage
    print("Setting up Student Dropout Prediction Model...")
    predictor = setup_model()
    
    if predictor:
        print("\nModel setup completed successfully!")
        print(f"Available models: {predictor.get_model_names()}")
        print(f"Number of features: {len(predictor.get_feature_names())}")
        
        # Example evaluation
        for model_name in predictor.get_model_names():
            metrics = predictor.evaluate_model(model_name)
            if metrics:
                print(f"\n{model_name} Performance:")
                for metric, value in metrics.items():
                    print(f"  {metric.title()}: {value:.4f}")
    else:
        print("Failed to setup model. Please check your dataset.")
