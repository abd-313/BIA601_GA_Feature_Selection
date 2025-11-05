// static/js/app.js

document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const dynamicForm = document.getElementById('dynamic-form');
    const sourceInput = document.getElementById('source_type');
    const formContainer = document.getElementById('dynamic-form-container');

    function loadFormTemplate(source) {
        if (sourceInput.value === source) {
            return;
        }

        sourceInput.value = source;
        
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
        
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set('source', source);
        window.location.href = currentUrl.toString();

    }

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

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const source = this.getAttribute('data-source');
            loadFormTemplate(source);
        });
    });

    // One day I will work with a team that actually can do all this stuff ... I hope at least lol
    // Learned few things with project I guess and it was fun
    //B_A
    const initialSource = sourceInput.value;
    if (initialSource) {
        updateTabStyles(initialSource);
    }
});