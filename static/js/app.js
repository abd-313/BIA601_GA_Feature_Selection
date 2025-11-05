// static/js/app.js

document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const dynamicForm = document.getElementById('dynamic-form');
    const sourceInput = document.getElementById('source_type');
    const formContainer = document.getElementById('dynamic-form-container');

    // Function to handle fetching and swapping form content
    function loadFormTemplate(source) {
        // Prevent loading if the source is already active
        if (sourceInput.value === source) {
            return;
        }

        // 1. Update Hidden Field for Django Backend
        sourceInput.value = source;
        
        // 2. Determine the correct template path to fetch (since we are on upload_dynamic)
        let templateName = '';
        if (source === 'repository') {
            templateName = 'repository_selection_form.html';
        } else if (source === 'upload_csv') {
            templateName = 'upload_csv_form.html';
        } else if (source === 'database') {
            templateName = 'database_link_form.html';
        } else {
            formContainer.innerHTML = '<div class="error-panel">Error: Invalid source type.</div>';
            return;
        }
        
        // 3. Simple AJAX call to fetch the form template (This assumes your view logic can return the form template content)
        // NOTE: In a real Django setup, this would typically involve an AJAX call to a dedicated view 
        // that renders the form, not fetching the template file directly. 
        // We'll simulate the update by changing the action attribute for the form based on the source 
        // or by simply displaying a message if not loaded via Django context.
        
        // Since the current context already loads the correct form via Django logic:
        // We use window.location to trigger a page refresh with the new GET parameter, which
        // reloads the 'upload_dynamic_view' with the correct form included.
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set('source', source);
        window.location.href = currentUrl.toString();

    }

    // Function to update button styles
    function updateTabStyles(activeSource) {
        tabButtons.forEach(button => {
            if (button.getAttribute('data-source') === activeSource) {
                button.classList.remove('inactive-tab-btn');
                button.classList.add('active-tab-btn');
            } else {
                button.classList.remove('active-tab-btn');
                button.classList.add('inactive-tab-btn');
            }
        });
    }

    // Event Listeners for Tab Buttons
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const source = this.getAttribute('data-source');
            loadFormTemplate(source);
        });
    });

    // Initialize the active tab style based on the initial Django context
    const initialSource = sourceInput.value;
    if (initialSource) {
        updateTabStyles(initialSource);
    }
});