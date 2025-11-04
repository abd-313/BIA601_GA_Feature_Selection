from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
# IMPORTANT: Assuming these forms (UploadCSVForm, DatabaseLinkForm, RepositoryForm) 
# are correctly defined in your analysis/forms.py file
from .forms import UploadCSVForm, DatabaseLinkForm, RepositoryForm 
import uuid # Used to generate a mock job ID for redirection

# --- Main Views ---

def home_view(request: HttpRequest) -> HttpResponse:
    """Renders the main menu for selecting the data input source."""
    # Template name changed to 'home.html' (short name)
    return render(request, 'home.html', {'title': 'Select Data Source'})


def upload_dynamic_view(request: HttpRequest) -> HttpResponse:
    """
    Handles the dynamic form display and submission for data input (CSV, DB, Repo).
    """
    source_type = request.GET.get('source', 'error')
    form = None
    title = "Error: Invalid Source"

    # 1. Determine the correct form and title based on the source parameter
    if source_type == 'upload_csv':
        form_class = UploadCSVForm
        title = "Upload Custom CSV for GA Optimization"
    elif source_type == 'repository':
        form_class = RepositoryForm
        title = "Select Dataset from Internal Repository"
    elif source_type == 'database':
        form_class = DatabaseLinkForm
        title = "Connect to External Database"
    else:
        # Error case: render the dynamic template with an error
        context = {'title': title, 'source_type': 'error'}
        return render(request, 'upload_dynamic.html', context)

    # 2. Handle Form Submission (POST Request)
    if request.method == 'POST':
        # Pass request.FILES for file uploads (CSV)
        form = form_class(request.POST, request.FILES) 
        if form.is_valid():
            
            # --- Mock GA Job Launch ---
            job_id = str(uuid.uuid4())
            
            # CRITICAL FIX: Redirect using the fully qualified named URL 'analysis:ga_status_view'
            # 'analysis' is the app_name defined in analysis/urls.py
            return redirect('analysis:ga_status_view', job_id=job_id)

    # 3. Handle Initial Form Display (GET Request)
    if form is None:
        form = form_class()

    context = {
        'title': title,
        'form': form,
        'source_type': source_type,
    }

    # Template name changed to 'upload_dynamic.html' (short name)
    return render(request, 'upload_dynamic.html', context)

def ga_status_view(request, job_id):
    """Renders the status page for a running or completed GA job."""
    
    # This function is now defined, resolving the view lookup error from urls.py
    context = {
        'title': f'GA Job Status: {job_id[:8]}...',
        'job_id': job_id,
        'status_message': "The Genetic Algorithm job is currently running...",
    }
    # Template name changed to 'ga_status.html' (short name)
    return render(request, 'ga_status.html', context)