# analysis/views.py
import uuid
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, Http404
import os
import joblib         # Required to load the .joblib files
import base64         # Required to encode the plot image
import json           # Used for handling best_params dictionary
import time           # Used for generating a simulated job ID
from io import BytesIO
from pathlib import Path
# --- [1] IMPORT REQUIRED FORMS ---
from .forms import UploadCSVForm, DatabaseLinkForm, RepositorySelectionForm 
from .ga_processor.ga_processor import process_ga_job

# --- Utility Functions ---
GA_JOB_RESULTS_ROOT = Path(__file__).parent / "ga_job_results"
def load_plot_image(file_path):
    """Loads a PNG image, encodes it to Base64, and returns a data URI string."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as image_file:
            plot_data_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{plot_data_base64}"
    except Exception as e:
        print(f"Error loading plot image {file_path}: {e}")
        return None

# --- View Functions ---

def home_view(request):
    """Displays the main menu of analysis options."""
    
    context = {
        'title': 'Genetic Algorithm Feature Selection Tool'
    }
    return render(request, 'home.html', context)


def display_repo_results(request):
    """Displays the pre-calculated repository results."""
    # Define directories relative to BASE_DIR (the project root)
    BASE_PROJECT_DIR = settings.BASE_DIR 
    DATA_DIR = os.path.join(BASE_PROJECT_DIR, 'data')
    TEMP_PLOTS_DIR = os.path.join(BASE_PROJECT_DIR, 'temp_plots')
    
    # Define file paths
    model_path = os.path.join(DATA_DIR, 'best_ga_model.joblib')
    plot_path = os.path.join(TEMP_PLOTS_DIR, 'fitness_evolution.png')
    
    best_score = "0.9250" 
    best_params_display = "No model loaded"
    plot_image_data = None
    
    try:
        # 1. Load Model and Extract Parameters
        if os.path.exists(model_path):
            best_model = joblib.load(model_path)
            
            # SAFE EXTRACTION: Use scikit-learn's standard get_params() method 
            if hasattr(best_model, 'get_params'):
                # Extract parameters used to train the model
                best_params_dict = best_model.get_params(deep=False) 
                
                # Exclude internal/technical parameters for cleaner display
                keys_to_exclude = ['warm_start', 'random_state', 'verbose', 'n_jobs', 'class_weight']
                clean_params = {k: v for k, v in best_params_dict.items() if k not in keys_to_exclude}
                best_params_display = json.dumps(clean_params, indent=2)
            else:
                best_params_display = "Loaded object is not a scikit-learn model."
            
            # 2. Load Plot Image
            plot_image_data = load_plot_image(plot_path)

        else:
            best_score = "CRITICAL ERROR: Model file not found in /data/"
            
    except Exception as e:
        print(f"CRITICAL ERROR loading analysis files: {e}")
        best_score = "Loading Error"
        best_params_display = f"An error occurred while loading files: {e}"

    context = {
        'dataset_name': 'Pre-analyzed Repository Dataset',
        'target_column': 'Target_Variable', 
        'status': 'Completed (Loaded from files)',
        'best_score': best_score,
        'best_params': best_params_display, 
        'plot_image_data': plot_image_data, # Passed to repository_results.html
    }
    
    return render(request, 'repository_results.html', context)


# VIEW: Handles dynamic form display (CSV, Database, Repository Selection)
# FIX: Guarantees that the 'form' object is created to prevent "Form Context Missing".
def upload_dynamic_view(request, preloaded_form=None, initial_source=None):
    # 1. Determine the active source from GET parameter or POST failure
    source = request.GET.get('source', 'repository') # Default to repository
    if initial_source:
        source = initial_source # Use source from failed POST request
        
    # 2. Set context variables for the template
    context = {'title': f'{source.replace("_", " ").title()} Configuration', 'source_type': source}
    
    # 3. Handle forms
    if source == 'upload_csv':
        form_class = UploadCSVForm
        context['form_template'] = 'upload_csv_form.html'
    elif source == 'database':
        form_class = DatabaseLinkForm
        context['form_template'] = 'database_link_form.html'
    elif source == 'repository':
        form_class = RepositorySelectionForm
        context['form_template'] = 'repository_selection_form.html'
    else:
        # Fallback in case of invalid source parameter
        return render(request, 'upload_dynamic.html', {'title': 'Error', 'form_template': None})

    # Load preloaded form (from failed POST) or create a new one
    context['form'] = preloaded_form if preloaded_form else form_class()
    
    return render(request, 'upload_dynamic.html', context)

# VIEW: Handles the form POST submission and redirects to results
def submit_dynamic_job(request):
    """
    Handles POST requests from all dynamic forms, processes data, runs the GA,
    and redirects to the results view.
    """
    if request.method == 'POST':
        source_type = request.POST.get('source_type')
        job_id = str(uuid.uuid4()) # Generate a unique ID for this job
        
        # 1. Instantiate the correct form
        if source_type == 'upload_csv':
            form = UploadCSVForm(request.POST, request.FILES)
        # Add other forms here if needed later
        # elif source_type == 'database':
        #     form = DatabaseLinkForm(request.POST) 
        # elif source_type == 'repository':
        #     form = RepositorySelectionForm(request.POST) 
        else:
            # Should not happen if tabs are used correctly
            return redirect('analysis:home_view')

        # 2. Validate the form
        if form.is_valid():
            
            # --- START: CSV UPLOAD LOGIC ---
            if source_type == 'upload_csv':
                # Get validated data
                csv_file = form.cleaned_data['csv_file']
                target_column = form.cleaned_data['target_column']
                model_choice = form.cleaned_data['model_choice']
                
                try:
                    # Read the uploaded file (InMemoryUploadedFile) directly into a Pandas DataFrame
                    # Using encoding='latin1' or 'ISO-8859-1' is sometimes safer for global CSV files
                    df = pd.read_csv(csv_file) 
                    
                    # 3. Run the Genetic Algorithm (GA) job
                    # This function executes the entire GA process in the background/foreground.
                    ga_results = process_ga_job(
                        input_data=df, 
                        target_column=target_column, 
                        model_choice=model_choice, 
                        job_id=job_id
                    )
                    
                    if ga_results.get('error'):
                        # If the GA job fails, attach the error to the form and return
                        print(f"GA Error: {ga_results['message']}")
                        form.add_error(None, f"GA Job failed: {ga_results['message']}")
                        # We must rely on your 'upload_dynamic_view' to handle rendering the form with errors
                        return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type) 
                    
                    # 4. Success: Redirect to the results page using the generated job_id
                    # Redirect uses the URL name 'repository_results_view'
                    return redirect('analysis:repository_results_view', job_id=job_id)
                
                except Exception as e:
                    # Handle general errors (e.g., file reading, Pandas issues)
                    print(f"Unexpected Error during CSV processing: {e}")
                    form.add_error(None, f"Unexpected error during file processing: {e}")
                    return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type)

            # --- END: CSV UPLOAD LOGIC ---
            
            # Implement logic for 'database' and 'repository' here later

        else:
            # If form validation fails, redirect back to the dynamic view with the failed form
            return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type)
    
    # If not a POST request, redirect home
    return redirect('analysis:home_view')


def repository_results_view(request, job_id):
    job_dir = GA_JOB_RESULTS_ROOT / job_id
    summary_file_path = job_dir / f'{job_id}_summary.json'
    
    if not summary_file_path.exists():
        raise Http404(f"Analysis results not found for Job ID: {job_id}. Ensure your CSV target column is categorical.")

    try:
        with open(summary_file_path, 'r') as f:
            results_data = json.load(f)
            
    except Exception as e:
        print(f"Error loading JSON summary for {job_id}: {e}")
        raise Http404(f"Error loading summary data for Job ID: {job_id}")

    plot_relative_url = f'/media/ga_jobs/{job_id}/fitness_evolution.png'

    context = {
        'job_id': job_id,
        'status': results_data.get('status', 'Completed'),
        'dataset_name': results_data.get('dataset_name', 'Uploaded CSV'),
        'target_column': results_data.get('target_column', 'N/A'),
        'ml_model': results_data.get('ml_model', 'Logistic Regression'),
        
        'ga_model_accuracy': f"{results_data.get('ga_model_accuracy', 0.0):.4f}",
        'ga_weighted_fitness': f"{results_data.get('ga_weighted_fitness', 0.0):.4f}",
        'n_features_selected': results_data.get('n_features_selected', 'N/A'),
        'total_features': results_data.get('total_features', 'N/A'),

        'baseline_results_markdown': results_data.get('baseline_results_markdown', 'No data.'),
        
        'plot_url': plot_relative_url, 
        
        'ga_params_json': json.dumps(results_data.get('ga_params', {}), indent=2),
    }

    return render(request, 'repository_results.html', context)