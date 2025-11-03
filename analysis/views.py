import os
import joblib
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# --- GLOBAL ML ARTIFACTS LOADING ---

# CRITICAL FIX: Define the project root by going up two levels from views.py:
# 1. .parent (analysis)
# 2. .parent (bia601_project)
# The parent of bia601_project should be the directory containing manage.py and data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the paths for the pre-trained model and feature mask
MODEL_PATH = PROJECT_ROOT / "data" / "best_ga_model.joblib"
MASK_PATH = PROJECT_ROOT / "data" / "best_feature_mask.joblib"

# Global variables to hold the loaded model and mask
GA_MODEL = None
FEATURE_MASK = None
N_FEATURES = 0
GA_ACCURACY = "N/A" # Placeholder for display

def load_ml_artifacts():
    """
    Loads the pre-trained GA model and feature mask into memory.
    """
    global GA_MODEL, FEATURE_MASK, N_FEATURES, GA_ACCURACY
    
    # Try to load the saved model and mask
    try:
        GA_MODEL = joblib.load(MODEL_PATH)
        FEATURE_MASK = joblib.load(MASK_PATH)
        
        # Calculate the number of selected features (True values in the mask)
        N_FEATURES = FEATURE_MASK.sum() 
        
        # Fixed accuracy value based on your final ML run results
        GA_ACCURACY = "0.9912" 
        
        print(f" ML Artifacts Loaded Successfully: Model={GA_MODEL.__class__.__name__}, Features={N_FEATURES}")
    except Exception as e:
        # Critical error handling
        print(f" ERROR: Failed to load ML artifacts from {MODEL_PATH} and {MASK_PATH}.")
        print(f"Error details: {e}")
        GA_MODEL = None
        FEATURE_MASK = None
        GA_ACCURACY = "0.000"

# Call the loading function immediately when the view module is imported
load_ml_artifacts()


def home_view(request):
    """
    View for the main landing page (/).
    """
    context = {
        'ga_accuracy': GA_ACCURACY,
        'features_selected': N_FEATURES,
        'model_name': GA_MODEL.__class__.__name__ if GA_MODEL else 'Not Loaded',
    }
    return render(request, 'home.html', context)


@csrf_exempt
def predict_static_api(request):
    """
    API endpoint for making a single, static prediction.
    """
    if request.method == 'POST':
        if GA_MODEL is None or FEATURE_MASK is None:
            return JsonResponse({'error': 'ML Model not loaded on the server. Check logs for missing data/joblib files.'}, status=500)
        
        # TODO: Real prediction logic will go here
        
        data = {'prediction': 'Activity (Placeholder)', 'status': 'success', 'accuracy': GA_ACCURACY}
        return JsonResponse(data)
    
    return JsonResponse({'error': 'Only POST method is allowed for this API endpoint.'}, status=405)


def upload_dynamic_view(request):
    """
    View for handling the long-running dynamic CSV upload.
    """
    if request.method == 'POST':
        # Placeholder response after file submission
        return HttpResponse("<h1>Your custom data analysis request has been queued (Future Feature).</h1>")
        
    return render(request, 'upload_dynamic.html')
