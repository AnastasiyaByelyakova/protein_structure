document.addEventListener('DOMContentLoaded', () => {
    console.log("main.js loaded and executing");

    // Get navigation links and sections
    const navLinks = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('main section');

    console.log("Selected navigation links:", navLinks);
    console.log("Identified sections on page load:", Array.from(sections).map(section => section.id));


    // Function to show a specific section and hide others
    const showSection = (sectionId) => {
        console.log("Attempting to show section:", sectionId);
        sections.forEach(section => {
            section.classList.add('hidden');
        });
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            console.log("Target section element:", targetSection);
            targetSection.classList.remove('hidden');
             // Scroll to the section smoothly
            targetSection.scrollIntoView({
                behavior: 'smooth'
            });
        } else {
            console.warn(`Target section with ID "${sectionId}" not found.`);
        }
    };

    // Handle navigation clicks
    navLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault(); // Prevent default link behavior
            console.log("Clicked link:", link);
            console.log("Link href:", link.href);
            const sectionId = link.getAttribute('href').substring(1); // Get section ID from href
            showSection(sectionId);
        });
    });

    // Initial state: show the auth section by default
    showSection('auth');

    // --- Form Submission Handlers ---

    // Get forms and result display areas
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const predictSequenceForm = document.getElementById('predict-sequence-form');
    const predictFileForm = document.getElementById('predict-file-form');
    const runValidationButton = document.getElementById('run-validation-button'); // Assuming a button for validation
    const retrainingForm = document.getElementById('retraining-form');


    const authResultDiv = document.getElementById('auth-result');
    const predictionResultDiv = document.getElementById('prediction-result');
    const modelInfoContentDiv = document.getElementById('model-info-content');
    const validationContentDiv = document.getElementById('validation-content');
    const retrainingResultDiv = document.getElementById('retraining-result');


    // Check if forms and buttons are found
    if (!loginForm) console.warn("Login form with ID 'login-form' not found.");
    if (!registerForm) console.warn("Registration form with ID 'register-form' not found.");
    if (!predictSequenceForm) console.warn("Predict sequence form with ID 'predict-sequence-form' not found.");
    if (!predictFileForm) console.warn("Predict file form with ID 'predict-file-form' not found.");
    if (!runValidationButton) console.warn("Initial Validation Trigger Button not found (might be added dynamically).");
    if (!retrainingForm) console.warn("Retraining form with ID 'retraining-form' not found.");


    // Helper function to handle fetch responses
    async function handleResponse(response) {
        if (!response.ok) {
            const error = await response.json();
            console.error(`Request failed with status ${response.status}:`, error);
            if (error.detail) {
                 throw new Error(`API Error: ${JSON.stringify(error.detail)}`);
            } else {
                 throw new Error(`API Error: ${response.statusText}`);
            }

        }
        return response.json();
    }

    // Handle Registration Form Submission
    if (registerForm) {
        registerForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Registration form submitted");

            const emailInput = registerForm.querySelector('#register-email');
            const passwordInput = registerForm.querySelector('#register-password');

            if (!emailInput) {console.error("Registration email input not found"); return;}
            if (!passwordInput) {console.error("Registration password input not found"); return;}

            const email = emailInput.value;
            const password = passwordInput.value;
            // Use email as username for registration
            const username = email;
            
            try {
                const response = await fetch('/register/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email, password, username }), // Include username here
                });

                console.log("Registration response status:", response.status);

                 if (!response.ok) {
                    const error = await response.json();
                    console.error(`Registration failed with status ${response.status}:`, error);
                     if (authResultDiv) {
                        if (error.detail) {
                            authResultDiv.innerHTML = `<p style="color: red;">Registration failed: ${JSON.stringify(error.detail)}</p>`;
                             console.log("Registration failed: Validation Error details:", error.detail);
                        } else {
                            authResultDiv.innerHTML = `<p style="color: red;">Registration failed: ${response.statusText}</p>`;
                        }
                    }
                    return; // Stop further processing on failure
                }


                const result = await response.json();
                console.log('Registration successful:', result);
                if (authResultDiv) {
                    authResultDiv.innerHTML = `<p style="color: green;">Registration successful! You can now log in.</p>`;
                }
                // Optionally clear the form or redirect to login
                 registerForm.reset();

            } catch (error) {
                console.error('Error during registration:', error);
                 if (authResultDiv) {
                    authResultDiv.innerHTML = `<p style="color: red;">An error occurred during registration.</p>`;
                }
            }
        });
    }

    // Handle Login Form Submission
    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Login form submitted");
            const emailInput = loginForm.querySelector('#login-email');
            const passwordInput = loginForm.querySelector('#login-password');

            if (!emailInput) {console.error("Login email input not found"); return;}
            if (!passwordInput) {console.error("Login password input not found"); return;}


            const email = emailInput.value;
            const password = passwordInput.value;


            try {
                // Assuming your login endpoint expects form-urlencoded data or JSON
                // Adjust the body and headers based on your FastAPI implementation
                const response = await fetch('/login/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json', // Or 'application/x-www-form-urlencoded'
                    },
                    body: JSON.stringify({ email: email, password }), // Adjust keys based on backend expectation
                });

                 if (!response.ok) {
                    const error = await response.json();
                    console.error(`Login failed with status ${response.status}:`, error);
                     if (authResultDiv) {
                        if (error.detail) {
                            authResultDiv.innerHTML = `<p style="color: red;">Login failed: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                            authResultDiv.innerHTML = `<p style="color: red;">Login failed: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }

                const result = await response.json();
                console.log('Login successful:', result);
                // Store token or user info (e.g., in localStorage)
                localStorage.setItem('access_token', result.access_token); // Adjust based on your backend response

                 if (authResultDiv) {
                    authResultDiv.innerHTML = `<p style="color: green;">Login successful! Redirecting...</p>`;
                }
                // Redirect or update UI to show other sections
                // For now, just log success and could potentially show other sections
                showSection('prediction'); // Example: automatically show prediction section after login


            } catch (error) {
                console.error('Error during login:', error);
                 if (authResultDiv) {
                    authResultDiv.innerHTML = `<p style="color: red;">An error occurred during login.</p>`;
                }
            }
        });
    }


    // Handle Predict Sequence Form Submission
    if (predictSequenceForm) {
        predictSequenceForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Predict sequence form submitted");
            const sequenceInput = predictSequenceForm.querySelector('#protein-sequence-input'); // Assuming the input is in the form
             if (!sequenceInput) {console.error("Sequence input not found"); return;}
            const sequence = sequenceInput.value;

             // Basic sequence validation (can be enhanced)
             const validProteinChars = /^[ACDEFGHIKLMNPQRSTUVWYXBZJOX]+$/i; // Includes common unknowns/modified
             if (!validProteinChars.test(sequence)) {
                 console.error("Invalid protein sequence:", sequence);
                 return;
             }


            try {
                const response = await fetch('/predict/', {
                    method: 'POST',
                    // Using FormData to send form data
                     headers: {
                         // 'Content-Type': 'multipart/form-data' is automatically set with FormData
                         'Authorization': `Bearer ${localStorage.getItem('access_token')}` // Include token if endpoint is authenticated
                    },
                    body: new URLSearchParams({ sequence: sequence }) // Send as x-www-form-urlencoded
                });

                 if (!response.ok) {
                    const error = await response.json();
                    console.error(`Prediction failed with status ${response.status}:`, error);
                     if (predictionResultDiv) {
                        if (error.detail) {
                            predictionResultDiv.innerHTML = `<p style="color: red;">Prediction failed: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                            predictionResultDiv.innerHTML = `<p style="color: red;">Prediction failed: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }

                const result = await response.json();
                console.log('Prediction successful:', result);
                 if (predictionResultDiv) {
                     // Display the predicted coordinates - you might want to format this nicely
                     predictionResultDiv.innerHTML = `<p style="color: green;">Prediction successful! Predicted Coordinates:</p><pre>${JSON.stringify(result, null, 2)}</pre>`;
                 }

            } catch (error) {
                console.error('Error during sequence prediction:', error);
                 if (predictionResultDiv) {
                    predictionResultDiv.innerHTML = `<p style="color: red;">An error occurred during sequence prediction.</p>`;
                }
            }
        });
    }

    // Handle Predict File Form Submission
    if (predictFileForm) {
         predictFileForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Predict file form submitted");
            const fileInput = predictFileForm.querySelector('#fasta-file-input');

            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                 if (predictionResultDiv) {
                     predictionResultDiv.innerHTML = `<p style="color: red;">Please select a FASTA file.</p>`;
                 }
                console.error("No file selected for prediction.");
                return;
            }

            const formData = new FormData();
            formData.append('fasta_file', fileInput.files[0]); // 'fasta_file' should match the parameter name in your FastAPI endpoint


            try {
                const response = await fetch('/predict/', { // Assuming the same endpoint handles file uploads
                    method: 'POST',
                     headers: {
                         // 'Content-Type': 'multipart/form-data' is automatically set with FormData
                         'Authorization': `Bearer ${localStorage.getItem('access_token')}` // Include token if endpoint is authenticated
                    },
                    body: formData,
                });


                if (!response.ok) {
                    const error = await response.json();
                    console.error(`File prediction failed with status ${response.status}:`, error);
                    if (predictionResultDiv) {
                         if (error.detail) {
                            predictionResultDiv.innerHTML = `<p style="color: red;">File prediction failed: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                            predictionResultDiv.innerHTML = `<p style="color: red;">File prediction failed: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }


                const result = await response.json();
                console.log('File prediction successful:', result);
                 if (predictionResultDiv) {
                      // Display the prediction results from the file - format depends on backend response
                     predictionResultDiv.innerHTML = `<p style="color: green;">File prediction successful! Results:</p><pre>${JSON.stringify(result, null, 2)}</pre>`;
                 }


            } catch (error) {
                console.error('Error during file prediction:', error);
                 if (predictionResultDiv) {
                    predictionResultDiv.innerHTML = `<p style="color: red;">An error occurred during file prediction.</p>`;
                }
            }
        });
    }


    // Handle Model Info Click (or page load for this section)
    // You might want to fetch model info when the 'model-info' section is shown
    const modelInfoLink = document.querySelector('a[href="#model-info"]');
    if (modelInfoLink) {
        modelInfoLink.addEventListener('click', async (event) => {
            // The showSection call handles the visibility, now fetch the data
            // event.preventDefault(); // showSection already prevents default
            console.log("Fetching model info...");
             if (modelInfoContentDiv) {
                modelInfoContentDiv.innerHTML = '<p>Loading model information...</p>';
            }

            try {
                const response = await fetch('/model_info/', { // Adjust endpoint if needed
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                    }
            });
                 if (!response.ok) {
                    const error = await response.json();
                    console.error(`Failed to fetch model info with status ${response.status}:`, error);
                    if (modelInfoContentDiv) {
                        if (error.detail) {
                            modelInfoContentDiv.innerHTML = `<p style="color: red;">Failed to load model info: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                             modelInfoContentDiv.innerHTML = `<p style="color: red;">Failed to load model info: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }

                const result = await response.json();
                console.log('Model info fetched:', result);
                 if (modelInfoContentDiv) {
                     // Format and display the model structure and parameters
                     modelInfoContentDiv.innerHTML = `
                         <h3>Model Structure:</h3>
                         <pre>${result.model_summary}</pre>
                         <h3>Model Parameters:</h3>
                         <pre>${JSON.stringify(result.last_validation_results, null, 2)}</pre>
                     `; // Adjust based on your backend response structure
                 }


            } catch (error) {
                console.error('Error fetching model info:', error);
                 if (modelInfoContentDiv) {
                    modelInfoContentDiv.innerHTML = `<p style="color: red;">An error occurred while fetching model information.</p>`;
                }
            }
        });
    }

    const validationLink = document.querySelector('a[href="#validation"]');
    if (validationLink) {
        validationLink.addEventListener('click', async (event) => {
            console.log("Fetching validation info...");
            if (validationContentDiv) {
                validationContentDiv.innerHTML = '<p>Loading validation information...</p>';
            }

            try {
                const response = await fetch('/validation/', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                    }
                });

                if (!response.ok) {
                    const error = await response.json();
                    console.error(`Failed to fetch validation info with status ${response.status}:`, error);
                    if (validationContentDiv) {
                        if (error.detail) {
                            validationContentDiv.innerHTML = `<p style="color: red;">Failed to load validation info: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                            validationContentDiv.innerHTML = `<p style="color: red;">Failed to load validation info: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }

                const result = await response.json();
                console.log('Validation info fetched:', result);
                if (validationContentDiv) {
                    validationContentDiv.innerHTML = `
                        <h3>Validation Results:</h3>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    `;
                }

            } catch (error) {
                console.error('Error fetching validation info:', error);
                if (validationContentDiv) {
                    validationContentDiv.innerHTML = `<p style="color: red;">An error occurred while fetching validation information.</p>`;
                }
            }
        });
    }


    // Handle Run Validation Button Click
    if (runValidationButton) {
        runValidationButton.addEventListener('click', async () => {
            console.log("Validation Triggered");
             if (validationContentDiv) {
                validationContentDiv.innerHTML = '<p>Running model validation... This may take a while.</p>';
            }

            try {
                const response = await fetch('/run_validation/', { // Adjust endpoint if needed
                    method: 'POST', // Assuming it's a POST to trigger an action
                     headers: {
                         'Authorization': `Bearer ${localStorage.getItem('access_token')}` // Include token if endpoint is authenticated
                    },
                    // Add body if validation requires input data
                });


                 if (!response.ok) {
                    const error = await response.json();
                    console.error(`Validation failed with status ${response.status}:`, error);
                     if (validationContentDiv) {
                         if (error.detail) {
                             validationContentDiv.innerHTML = `<p style="color: red;">Validation failed: ${JSON.stringify(error.detail)}</p>`;
                         } else {
                            validationContentDiv.innerHTML = `<p style="color: red;">Validation failed: ${response.statusText}</p>`;
                         }
                    }
                    return;
                }


                const result = await response.json();
                console.log('Validation successful:', result);
                 if (validationContentDiv) {
                     // Display the validation results
                     validationContentDiv.innerHTML = `
                         <p style="color: green;">Validation completed!</p>
                         <h3>Validation Results:</h3>
                         <pre>${JSON.stringify(result.results, null, 2)}</pre>
                     `; // Adjust based on your backend response structure
                 }


            } catch (error) {
                console.error('Error running validation:', error);
                 if (validationContentDiv) {
                    validationContentDiv.innerHTML = `<p style="color: red;">An error occurred during validation.</p>`;
                }
            }
        });
    }


    // Handle Retraining Form Submission
    if (retrainingForm) {
        retrainingForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Retraining form submitted");

            if (retrainingResultDiv) {
                retrainingResultDiv.innerHTML = '<p>Starting model retraining... This may take a significant amount of time.</p>';
            }

            const formData = new FormData(retrainingForm);

            try {
                const response = await fetch('/retrain_model/', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                    },
                    body: formData,
                });

                if (!response.ok) {
                    const error = await response.json();
                    console.error(`Retraining failed with status ${response.status}:`, error);
                    if (retrainingResultDiv) {
                        if (error.detail) {
                            retrainingResultDiv.innerHTML = `<p style="color: red;">Retraining failed: ${JSON.stringify(error.detail)}</p>`;
                        } else {
                            retrainingResultDiv.innerHTML = `<p style="color: red;">Retraining failed: ${response.statusText}</p>`;
                        }
                    }
                    return;
                }

                const result = await response.json();
                console.log('Retraining successful:', result);
                if (retrainingResultDiv) {
                    retrainingResultDiv.innerHTML = `
                        <p style="color: green;">Retraining completed!</p>
                        <h3>Retraining Results:</h3>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    `;
                }

            } catch (error) {
                console.error('Error running retraining:', error);
                if (retrainingResultDiv) {
                    retrainingResultDiv.innerHTML = `<p style="color: red;">An error occurred during retraining.</p>`;
                }
            }
        });
    }

});