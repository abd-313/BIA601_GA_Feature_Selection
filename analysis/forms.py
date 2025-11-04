from django import forms

# --- 1. Form for CSV File Upload ---
class UploadCSVForm(forms.Form):
    """
    Form used for uploading a custom CSV file.
    """
    # FileField for the CSV file itself
    csv_file = forms.FileField(
        label='Select CSV File',
        help_text='Only .csv files are supported for custom upload.',
        widget=forms.FileInput(attrs={
            'accept': '.csv',
            'class': 'file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100 cursor-pointer w-full'
        })
    )
    
    # Optional field for the target column name
    target_column_name = forms.CharField(
        label='Target Variable Name (Optional)',
        required=False,
        max_length=100,
        help_text='If left blank, the last column will be assumed as the target.',
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., Target, Price, Activity',
            'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'
        })
    )

# --- 2. Form for Database Connection Parameters ---
class DatabaseLinkForm(forms.Form):
    """
    Form used for connecting to an external database (Placeholder).
    """
    db_host = forms.CharField(
        label='Database Host/IP',
        max_length=255,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., localhost:5432 or 192.168.1.1', 'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )
    db_name = forms.CharField(
        label='Database Name',
        max_length=100,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., project_db', 'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )
    db_user = forms.CharField(
        label='Username',
        max_length=100,
        widget=forms.TextInput(attrs={'placeholder': 'e.g., dbuser', 'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )
    db_password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )

# --- 3. Form for Internal Repository Data Selection (Mock) ---
class RepositoryForm(forms.Form):
    """
    Form used for selecting data from a pre-defined internal repository (Mockup).
    """
    # Mock choices for data selection
    DATA_CHOICES = [
        ('mock_sales', 'Mock Sales Data (100k rows)'),
        ('mock_chem', 'Mock Chemical Properties (50 features)'),
        ('mock_finance', 'Mock Financial Transactions (20 features)'),
    ]
    
    data_source = forms.ChoiceField(
        label='Select Repository Dataset',
        choices=DATA_CHOICES,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )

    # Optional GA configuration fields (shared by all data sources eventually)
    num_generations = forms.IntegerField(
        label='GA Generations',
        min_value=1,
        initial=50,
        help_text='Number of evolutionary cycles for the algorithm.',
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500'})
    )
