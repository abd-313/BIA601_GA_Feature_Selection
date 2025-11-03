/* app.js */

document.addEventListener('DOMContentLoaded', () => {
    console.log("BIA601 Application Frontend Initialized.");

    const staticForm = document.getElementById('static-form');
    const dynamicForm = document.getElementById('dynamic-form');
    const statusMessage = document.getElementById('status-message');

    /**
     * Shows a message in the status area.
     * @param {string} message - The message text.
     * @param {string} type - 'success', 'error', or 'info'.
     */
    function showStatus(message, type = 'info') {
        if (!statusMessage) return;

        // Reset classes
        statusMessage.className = 'mt-4 p-3 rounded-lg font-medium text-center';

        switch (type) {
            case 'success':
                statusMessage.classList.add('bg-green-100', 'text-green-800');
                break;
            case 'error':
                statusMessage.classList.add('bg-red-100', 'text-red-800');
                break;
            case 'info':
            default:
                statusMessage.classList.add('bg-blue-100', 'text-blue-800');
                break;
        }
        statusMessage.textContent = message;
        statusMessage.style.display = 'block';
    }

    /**
     * Handles the static prediction form submission (AJAX).
     * @param {Event} e 
     */
    if (staticForm) {
        staticForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitButton = staticForm.querySelector('button[type="submit"]');
            
            showStatus("Processing static prediction...", 'info');
            submitButton.disabled = true;
            submitButton.textContent = "Processing...";

            try {
                // In a real scenario, we would collect form data here.
                const response = await fetch('/predict/static/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        // Include CSRF token if not using @csrf_exempt
                        'X-CSRFToken': getCookie('csrftoken') 
                    },
                    // Send mock data for placeholder
                    body: JSON.stringify({ mock: true }) 
                });

                const data = await response.json();

                if (response.ok) {
                    showStatus(`Prediction Result: ${data.prediction} (Accuracy: ${data.accuracy})`, 'success');
                } else {
                    showStatus(`API Error: ${data.error || 'Server rejected the request.'}`, 'error');
                }

            } catch (error) {
                console.error("Fetch error:", error);
                showStatus("Network Error: Could not connect to the server.", 'error');
            } finally {
                submitButton.disabled = false;
                submitButton.textContent = "Go to Static Analysis";
            }
        });
    }

    /**
     * Handles the dynamic upload form submission.
     * Note: This is designed for standard form submission as the process is long-running.
     * The response will be an HTML page showing the job status.
     */
    if (dynamicForm) {
        dynamicForm.addEventListener('submit', (e) => {
            // For long-running tasks, we let the form submit normally
            // but we can show a temporary loading message.
            const submitButton = dynamicForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.textContent = "Starting GA Job...";
        });
    }

    /**
     * Helper function to get the CSRF token from cookies.
     * @param {string} name 
     * @returns {string | null}
     */
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.startsWith(name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});

