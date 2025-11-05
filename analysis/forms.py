from django import forms

MODEL_CHOICES = [
    ('rf', 'Random Forest Classifier'),
    ('svm', 'Support Vector Machine (Linear Kernel)'),
    ('lr', 'Logistic Regression'),
]

REPO_DATASET_CHOICES = [
    ('mock_sales', 'Mock Sales Data (Repository 1)'),
    ('mock_finance', 'Mock Financial Transactions (Repository 2)'),
]

INPUT_WIDGET = forms.TextInput(attrs={'class': 'input-field', 'placeholder': 'e.g., target_variable'})
SELECT_WIDGET = forms.Select(attrs={'class': 'input-field'})

class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(
        label='Select CSV File',
        help_text='Only .csv files are supported.',
        widget=forms.FileInput(attrs={'accept': '.csv', 'class': 'file-input-style'})
    )
    
    target_column = forms.CharField(
        label='Target Column Name',
        max_length=100,
        help_text='The name of the column you want to predict.',
        widget=INPUT_WIDGET
    )

    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Machine Learning Model',
        widget=SELECT_WIDGET
    )


class DatabaseLinkForm(forms.Form):

    db_url = forms.CharField(
        label='Database Connection URL',
        max_length=500,
        help_text='Full connection string (e.g., postgresql://user:pass@host:port/dbname).',
        widget=INPUT_WIDGET
    )
    
    query = forms.CharField(
        label='SQL Query (Optional)',
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'input-field', 'placeholder': 'e.g., SELECT * FROM your_table'})
    )

    target_column = forms.CharField(
        label='Target Column Name',
        max_length=100,
        widget=INPUT_WIDGET
    )

    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Machine Learning Model',
        widget=SELECT_WIDGET
    )


class RepositorySelectionForm(forms.Form):

    repository_dataset = forms.ChoiceField(
        choices=REPO_DATASET_CHOICES,
        label='Select Dataset',
        widget=SELECT_WIDGET
    )
    
    target_column = forms.CharField(
        label='Target Column Name',
        max_length=100,
        widget=INPUT_WIDGET
    )
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Machine Learning Model',
        widget=SELECT_WIDGET
    )

