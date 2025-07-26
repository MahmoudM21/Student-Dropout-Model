import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from dropout_model import StudentDropoutPredictor
import threading

class StudentDropoutPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team BMawy 😎 - Machine Learning Project - Student Dropout Prediction")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize predictor
        self.predictor = StudentDropoutPredictor()
        self.feature_entries = {}
        self.feature_names = []
        self.is_model_loaded = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Header section
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame, 
            text="Team BMawy 😎  - Machine Learning Project", 
            bg="#2c3e50", 
            fg="white", 
            font=("Arial", 24, "bold")
        )
        header_label.pack(expand=True)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left sidebar
        self.create_sidebar(main_container)
        
        # Right main area
        self.create_main_area(main_container)
        
    def create_sidebar(self, parent):
        sidebar_frame = tk.Frame(parent, bg="#34495e", width=250)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_frame.pack_propagate(False)
        
        # Sidebar title
        sidebar_title = tk.Label(
            sidebar_frame, 
            text="ML Projects", 
            bg="#34495e", 
            fg="white", 
            font=("Arial", 18, "bold")
        )
        sidebar_title.pack(pady=20)
        
        # Project buttons
        projects = [
            "Student Dropout Prediction",
            "Linear Regression", 
            "Classification Models",
            "Data Visualization",
            "Feature Engineering",
            "Model Evaluation",
            "Deep Learning",
            
        ]
        
        for i, project in enumerate(projects):
            btn_color = "#e74c3c" if i == 0 else "#7f8c8d"
            text_color = "white"
            
            project_btn = tk.Button(
                sidebar_frame,
                text=project,
                bg=btn_color,
                fg=text_color,
                font=("Arial", 12),
                relief=tk.FLAT,
                padx=15,
                pady=10,
                anchor="w"
            )
            project_btn.pack(fill=tk.X, padx=15, pady=5)
            
    def create_main_area(self, parent):
        main_frame = tk.Frame(parent, bg='#f0f0f0')
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Student Dropout Prediction System",
            font=("Arial", 28, "bold"),
            bg='#f0f0f0',
            fg="#2c3e50"
        )
        title_label.pack(pady=20)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Input form
        self.create_input_form(main_frame)
        
        # Results section
        self.create_results_section(main_frame)
        
    def create_control_panel(self, parent):
        control_frame = tk.LabelFrame(
            parent, 
            text="Model Control", 
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg="#2c3e50",
            padx=10,
            pady=10
        )
        control_frame.pack(fill=tk.X, pady=10)
        
        # Load dataset button
        load_btn = tk.Button(
            control_frame,
            text="Load Dataset & Train Models",
            command=self.load_and_train_models,
            bg="#3498db",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Model selection
        tk.Label(
            control_frame, 
            text="Select Model:", 
            font=("Arial", 12),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT, padx=(20, 5))
        
        self.model_var = tk.StringVar(value="Decision Tree")
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["Decision Tree", "Logistic Regression", "SVM"],
            state="readonly",
            font=("Arial", 12)
        )
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Please load dataset first",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg="#e74c3c"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
    def create_input_form(self, parent):
        # Create scrollable frame for inputs
        canvas = tk.Canvas(parent, bg='#f0f0f0', height=300)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input form frame
        form_frame = tk.LabelFrame(
            parent,
            text="Student Information Input",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg="#2c3e50",
            padx=10,
            pady=10
        )
        form_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initially show placeholder
        self.show_input_placeholder()
        
    def show_input_placeholder(self):
        placeholder_label = tk.Label(
            self.scrollable_frame,
            text="Load dataset first to see input fields",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg="#7f8c8d"
        )
        placeholder_label.pack(expand=True, pady=50)
        
    def create_input_fields(self):
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.feature_entries = {}
        
        # Create input fields in a grid
        row = 0
        col = 0
        max_cols = 3
        
        for feature in self.feature_names:
            # Create frame for each input
            input_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
            input_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            # Label
            label = tk.Label(
                input_frame,
                text=feature.replace('_', ' ').title() + ":",
                font=("Arial", 10),
                bg='#f0f0f0',
                anchor="w"
            )
            label.pack(fill=tk.X)
            
            # Entry
            entry = tk.Entry(
                input_frame,
                font=("Arial", 10),
                width=15
            )
            entry.pack(fill=tk.X, pady=(2, 0))
            entry.insert(0, "0")  # Default value
            
            self.feature_entries[feature] = entry
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        # Configure grid weights
        for i in range(max_cols):
            self.scrollable_frame.grid_columnconfigure(i, weight=1)
            
        # Predict button
        predict_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
        predict_frame.grid(row=row+1, column=0, columnspan=max_cols, pady=20)
        
        predict_btn = tk.Button(
            predict_frame,
            text="Predict Dropout Risk",
            command=self.predict_dropout,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=10
        )
        predict_btn.pack()
        
        # Clear all button
        clear_btn = tk.Button(
            predict_frame,
            text="Clear All Fields",
            command=self.clear_all_fields,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 12),
            padx=20,
            pady=8
        )
        clear_btn.pack(pady=(10, 0))
        
    def create_results_section(self, parent):
        results_frame = tk.LabelFrame(
            parent,
            text="Prediction Results",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg="#2c3e50",
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = tk.Label(
            results_frame,
            text="Enter student information and click 'Predict' to see results",
            font=("Arial", 16),
            bg='#f0f0f0',
            fg="#7f8c8d",
            wraplength=800
        )
        self.result_label.pack(pady=20)
        
    def load_and_train_models(self):
        """Load dataset and train models in a separate thread"""
        def load_task():
            try:
                # Ask for dataset file
                file_path = filedialog.askopenfilename(
                    title="Select Dataset CSV File",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                
                if not file_path:
                    return
                    
                self.status_label.config(text="Loading dataset...", fg="#f39c12")
                self.root.update()
                
                # Initialize predictor with selected file
                self.predictor = StudentDropoutPredictor(file_path)
                
                # Load and preprocess data
                if not self.predictor.load_and_preprocess_data():
                    raise Exception("Failed to load data")
                    
                self.status_label.config(text="Preparing data...", fg="#f39c12")
                self.root.update()
                
                if not self.predictor.prepare_data():
                    raise Exception("Failed to prepare data")
                    
                self.status_label.config(text="Training models...", fg="#f39c12")
                self.root.update()
                
                if not self.predictor.train_models():
                    raise Exception("Failed to train models")
                    
                # Get feature names and create input fields
                self.feature_names = self.predictor.get_feature_names()
                self.is_model_loaded = True
                
                # Update UI in main thread
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self.status_label.config(text=error_msg, fg="#e74c3c"))
                
        # Start loading in background thread
        threading.Thread(target=load_task, daemon=True).start()
        
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.status_label.config(text="Models trained successfully!", fg="#27ae60")
        self.create_input_fields()
        
    def predict_dropout(self):
        """Make dropout prediction"""
        if not self.is_model_loaded:
            messagebox.showerror("Error", "Please load and train models first!")
            return
            
        try:
            # Collect input values
            input_values = []
            for feature in self.feature_names:
                value = self.feature_entries[feature].get().strip()
                if not value:
                    value = "0"
                input_values.append(float(value))
                
            # Make prediction
            model_name = self.model_var.get()
            prediction, probability = self.predictor.predict_dropout(input_values, model_name)
            
            if prediction is None:
                self.result_label.config(
                    text=f"Error making prediction: {probability}",
                    fg="#e74c3c"
                )
                return
                
            # Display results
            if prediction == 1:
                risk_level = "HIGH RISK" if probability > 0.7 else "MODERATE RISK"
                color = "#e74c3c" if probability > 0.7 else "#f39c12"
                result_text = f"🚨 DROPOUT PREDICTION: {risk_level}\n"
                result_text += f"Dropout Probability: {probability:.2%}\n"
                result_text += f"Model Used: {model_name}"
            else:
                result_text = f"✅ LOW DROPOUT RISK\n"
                result_text += f"Dropout Probability: {probability:.2%}\n"
                result_text += f"Model Used: {model_name}"
                color = "#27ae60"
                
            self.result_label.config(text=result_text, fg=color)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields!")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
            
    def clear_all_fields(self):
        """Clear all input fields"""
        for entry in self.feature_entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, "0")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StudentDropoutPredictionApp(root)
    root.mainloop()