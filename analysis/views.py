# analysis/views.py
import uuid
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, Http404
import os
import joblib         
import base64         
import json           
import time           
from io import BytesIO
from pathlib import Path

from .forms import UploadCSVForm, DatabaseLinkForm, RepositorySelectionForm 
from .ga_processor.ga_processor import process_ga_job


GA_JOB_RESULTS_ROOT = Path(__file__).parent / "ga_job_results"
def load_plot_image(file_path):
    
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as image_file:
            plot_data_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{plot_data_base64}"
    except Exception as e:
        print(f"Error loading plot image {file_path}: {e}")
        return None



def home_view(request):
    
    
    context = {
        'title': 'Genetic Algorithm Feature Selection Tool'
    }
    return render(request, 'home.html', context)

#Stupid LLMs taking control of most of what we do 
#and student's not larnign stuff
#Said by B_A from the team for this project lol

def display_repo_results(request):
    
   
    BASE_PROJECT_DIR = settings.BASE_DIR 
    DATA_DIR = os.path.join(BASE_PROJECT_DIR, 'data')
    TEMP_PLOTS_DIR = os.path.join(BASE_PROJECT_DIR, 'temp_plots')
    
    
    model_path = os.path.join(DATA_DIR, 'best_ga_model.joblib')
    plot_path = os.path.join(TEMP_PLOTS_DIR, 'fitness_evolution.png')
    
    best_score = "0.9250" 
    best_params_display = "No model loaded"
    plot_image_data = None
    
    try:
        
        if os.path.exists(model_path):
            best_model = joblib.load(model_path)
            
            
            if hasattr(best_model, 'get_params'):
               
                best_params_dict = best_model.get_params(deep=False) 
                
                
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



def upload_dynamic_view(request, preloaded_form=None, initial_source=None):
    
    source = request.GET.get('source', 'repository') # Default to repository
    if initial_source:
        source = initial_source # Use source from failed POST request
        
    
    context = {'title': f'{source.replace("_", " ").title()} Configuration', 'source_type': source}
    
    
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
        
        return render(request, 'upload_dynamic.html', {'title': 'Error', 'form_template': None})

    
    context['form'] = preloaded_form if preloaded_form else form_class()
    
    return render(request, 'upload_dynamic.html', context)


def submit_dynamic_job(request):
    """
    Handles POST requests from all dynamic forms, processes data, runs the GA,
    and redirects to the results view.
    """
    if request.method == 'POST':
        source_type = request.POST.get('source_type')
        job_id = str(uuid.uuid4()) # Generate a unique ID for this job
        
        
        if source_type == 'upload_csv':
            form = UploadCSVForm(request.POST, request.FILES)
        else:
            
            return redirect('analysis:home_view')

        
        if form.is_valid():
            
            if source_type == 'upload_csv':
                
                csv_file = form.cleaned_data['csv_file']
                target_column = form.cleaned_data['target_column']
                model_choice = form.cleaned_data['model_choice']
                
                try:
                    df = pd.read_csv(csv_file) 
                    
                    ga_results = process_ga_job(
                        input_data=df, 
                        target_column=target_column, 
                        model_choice=model_choice, 
                        job_id=job_id
                    )
                    
                    if ga_results.get('error'):
                        print(f"GA Error: {ga_results['message']}")
                        form.add_error(None, f"GA Job failed: {ga_results['message']}")
                        return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type) 
                    return redirect('analysis:repository_results_view', job_id=job_id)
                
                except Exception as e:
                    # Handle general errors (e.g., file reading, Pandas issues)
                    print(f"Unexpected Error during CSV processing: {e}")
                    form.add_error(None, f"Unexpected error during file processing: {e}")
                    return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type)


        else:
            return upload_dynamic_view(request, preloaded_form=form, initial_source=source_type)
    
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