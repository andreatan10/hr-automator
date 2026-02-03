import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
import pickle
import os

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class RoleClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, df):
        """Combine education, experience, skills into single text"""
        texts = []
        for _, row in df.iterrows():
            education = str(row.get('Education', '')) if pd.notna(row.get('Education', '')) else ''
            experience = str(row.get('Experience', '')) if pd.notna(row.get('Experience', '')) else ''
            skills = str(row.get('Skills', '')) if pd.notna(row.get('Skills', '')) else ''
            
            text = f"Education: {education}. Experience: {experience}. Skills: {skills}"
            texts.append(text)
        
        return texts
    
    def train(self, df, test_size=0.2, val_size=0.1, epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the DistilBERT model with early stopping"""
        # Prepare data
        texts = self.prepare_data(df)
        labels = self.label_encoder.fit_transform(df['Role'])
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        
        if min_count < 2:
            print(f"Warning: Some classes have only {min_count} sample(s). Using random split instead of stratified.")
            # Use random split without stratification
            X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42)
        else:
            # Use stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=test_size, stratify=labels, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, random_state=42)
        
        # Store test data for later use
        self.test_texts = X_test
        self.test_labels = y_test
        
        # Create datasets
        train_dataset = ResumeDataset(X_train, y_train, self.tokenizer)
        val_dataset = ResumeDataset(X_val, y_val, self.tokenizer)
        test_dataset = ResumeDataset(X_test, y_test, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        num_labels = len(self.label_encoder.classes_)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Early stopping variables
        best_val_score = 0
        patience = 2
        patience_counter = 0
        
        # Training loop with early stopping
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            start_time = time.time()
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                
                # Progress tracking every 100 samples
                samples_processed = batch_count * batch_size
                if samples_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}: Processed {samples_processed} samples in {elapsed:.1f}s")
            
            # Validation with multiple metrics
            val_metrics = self.evaluate_with_metrics(val_loader)
            val_score = val_metrics['combined_score']
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
            
            # Early stopping check
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Test evaluation
        test_metrics = self.evaluate_with_metrics(test_loader)
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
        
        # Detailed evaluation
        self.detailed_evaluation(test_loader)
        
        # Save test predictions to Excel
        self.save_test_predictions()
        
        return self.model
    
    def evaluate_with_metrics(self, dataloader):
        """Evaluate model with multiple metrics"""
        self.model.eval()
        predictions, true_labels, probabilities = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        try:
            roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
            pr_auc = average_precision_score(true_labels, probabilities, average='macro')
        except:
            roc_auc = pr_auc = 0.0
        
        # Combined score
        combined_score = 0.5 * accuracy + 0.25 * roc_auc + 0.25 * pr_auc
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'combined_score': combined_score
        }
    
    def detailed_evaluation(self, dataloader):
        """Detailed evaluation with classification report"""
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Convert back to role names
        true_roles = self.label_encoder.inverse_transform(true_labels)
        pred_roles = self.label_encoder.inverse_transform(predictions)
        
        print("\nClassification Report:")
        print(classification_report(true_roles, pred_roles))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_roles, pred_roles))
    
    def predict(self, text, top_k=3):
        """Predict role for new resume text"""
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                role = self.label_encoder.inverse_transform([top_indices[0][i].cpu().numpy()])[0]
                confidence = top_probs[0][i].cpu().numpy()
                results.append({'role': role, 'confidence': float(confidence)})
        
        return results
    def save_test_predictions(self):
        """Save test dataset predictions to Excel"""
        results = []
        
        for i, text in enumerate(self.test_texts):
            predictions = self.predict(text, top_k=1)
            actual_role = self.label_encoder.inverse_transform([self.test_labels[i]])[0]
            
            # Parse text to extract components
            parts = text.split('. ')
            education = parts[0].replace('Education: ', '') if len(parts) > 0 else ''
            experience = parts[1].replace('Experience: ', '') if len(parts) > 1 else ''
            skills = parts[2].replace('Skills: ', '') if len(parts) > 2 else ''
            
            results.append({
                'Actual_Role': actual_role,
                'Predicted_Role': predictions[0]['role'],
                'Confidence': predictions[0]['confidence'],
                'Education': education,
                'Experience': experience,
                'Skills': skills
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_excel('test_predictions.xlsx', index=False)
        print(f"Test predictions saved to test_predictions.xlsx")
    
    def save_model(self, path='models/role_classifier'):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        with open(f'{path}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/role_classifier'):
        """Load saved model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        
        with open(f'{path}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Model loaded from {path}")

def main():
    # Check if model exists
    if os.path.exists('models/role_classifier'):
        print("Found existing trained model. Loading...")
        classifier = RoleClassifier()
        classifier.load_model()
        
        # Load data for predictions
        try:
            df = pd.read_excel('extracted_resume_data.xlsx')
            df_filtered = df[df['Name'] != 'invalid'].copy()
            df_filtered['Role'] = df_filtered['Role'].str.lower()
            
            # Make predictions on 20% subset of data
            df_subset = df_filtered.sample(frac=0.2, random_state=42)
            print(f"Making predictions on {len(df_subset)} resumes (20% subset)")
            
            results = []
            for idx, row in df_subset.iterrows():
                education = str(row.get('Education', '')) if pd.notna(row.get('Education', '')) else ''
                experience = str(row.get('Experience', '')) if pd.notna(row.get('Experience', '')) else ''
                skills = str(row.get('Skills', '')) if pd.notna(row.get('Skills', '')) else ''
                
                text = f"Education: {education}. Experience: {experience}. Skills: {skills}"
                predictions = classifier.predict(text, top_k=1)
                
                results.append({
                    'Name': row.get('Name', ''),
                    'Actual_Role': row.get('Role', ''),
                    'Predicted_Role': predictions[0]['role'],
                    'Confidence': predictions[0]['confidence'],
                    'Education': education,
                    'Experience': experience,
                    'Skills': skills
                })
            
            results_df = pd.DataFrame(results)
            results_df.to_excel('loaded_model_predictions.xlsx', index=False)
            print(f"Predictions saved to loaded_model_predictions.xlsx")
            return
        except FileNotFoundError:
            print("Data file not found for predictions")
            return
    
    # Train new model if none exists
    print("No existing model found. Training new model...")
    try:
        df = pd.read_excel('extracted_resume_data.xlsx')
    except FileNotFoundError:
        print("Please run resume_extractor.py first to generate extracted_resume_data.xlsx")
        return
    
    print(f"Loaded {len(df)} resumes")
    
    df_filtered = df[df['Name'] != 'invalid'].copy()
    print(f"After filtering invalid names: {len(df_filtered)} resumes")
    
    # Filter for selected candidates only
    df_filtered['Decision'] = df_filtered['Decision'].str.lower()
    df_selected = df_filtered[df_filtered['Decision'] == 'select'].copy()
    print(f"After filtering for selected candidates: {len(df_selected)} resumes")
    
    # Standardize all roles to lowercase
    df_selected['Role'] = df_selected['Role'].str.lower()
    
    # Keep only roles with at least 3 samples
    role_counts = df_selected['Role'].value_counts()
    valid_roles = role_counts[role_counts >= 3].index
    df_final = df_selected[df_selected['Role'].isin(valid_roles)]
    
    print(f"After filtering roles with <3 samples: {len(df_final)} resumes")
    print(f"Final roles: {df_final['Role'].value_counts()}")
    
    if len(df_final) < 10:
        print("Not enough data to train. Need at least 10 samples with 3+ samples per class.")
        return
    
    # Initialize and train classifier
    classifier = RoleClassifier()
    classifier.train(df_final, epochs=5, batch_size=8, learning_rate=3e-5)
    
    # Save model
    classifier.save_model()

if __name__ == "__main__":
    main()