document.addEventListener('DOMContentLoaded', () => {
    console.log("main.js loaded and executing");

    const navLinks = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('main section');

    console.log("Selected navigation links:", navLinks);
    console.log("Identified sections on page load:", Array.from(sections).map(section => section.id));

    const showSection = (sectionId) => {
        console.log("Attempting to show section:", sectionId);
        sections.forEach(section => {
            section.classList.add('hidden');
        });
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            console.log("Target section element:", targetSection);
            targetSection.classList.remove('hidden');
            targetSection.scrollIntoView({
                behavior: 'smooth'
            });
        } else {
            console.warn(`Target section with ID "${sectionId}" not found.`);
        }
    };

    navLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            console.log("Clicked link:", link);
            console.log("Link href:", link.href);
            const sectionId = link.getAttribute('href').substring(1);
            showSection(sectionId);
        });
    });

    showSection('prediction');

    const predictSequenceForm = document.getElementById('predict-sequence-form');
    const predictFileForm = document.getElementById('predict-file-form');
    const runValidationButton = document.getElementById('run-validation-button');
    const retrainingForm = document.getElementById('retraining-form');

    const predictionResultDiv = document.getElementById('prediction-result');
    const modelInfoContentDiv = document.getElementById('model-info-content');
    const validationContentDiv = document.getElementById('validation-content');
    const retrainingResultDiv = document.getElementById('retraining-result');

    if (!predictSequenceForm) console.warn("Predict sequence form with ID 'predict-sequence-form' not found.");
    if (!predictFileForm) console.warn("Predict file form with ID 'predict-file-form' not found.");
    if (!runValidationButton) console.warn("Initial Validation Trigger Button not found (might be added dynamically).");
    if (!retrainingForm) console.warn("Retraining form with ID 'retraining-form' not found.");

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

    if (predictSequenceForm) {
        predictSequenceForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Predict sequence form submitted");
            const sequenceInput = predictSequenceForm.querySelector('#protein-sequence-input');
             if (!sequenceInput) {console.error("Sequence input not found"); return;}
            const sequence = sequenceInput.value;

             const validProteinChars = /^[ACDEFGHIKLMNPQRSTUVWYXBZJOX]+$/i;
             if (!validProteinChars.test(sequence)) {
                 console.error("Invalid protein sequence:", sequence);
                 return;
             }

            try {
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: new URLSearchParams({ sequence: sequence })
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
            formData.append('fasta_file', fileInput.files[0]);

            try {
                const response = await fetch('/predict/', {
                    method: 'POST',
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

    const modelInfoLink = document.querySelector('a[href="#model-info"]');
    if (modelInfoLink) {
        modelInfoLink.addEventListener('click', async (event) => {
            console.log("Fetching model info...");
             if (modelInfoContentDiv) {
                modelInfoContentDiv.innerHTML = '<p>Loading model information...</p>';
            }

            try {
                const response = await fetch('/model_info/');
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
                     modelInfoContentDiv.innerHTML = `
                         <h3>Model Structure:</h3>
                         <pre>${result.model_summary}</pre>
                         <h3>Model Parameters:</h3>
                         <pre>${JSON.stringify(result.last_validation_results, null, 2)}</pre>
                     `;
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
                const response = await fetch('/validation/');

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

    if (runValidationButton) {
        runValidationButton.addEventListener('click', async () => {
            console.log("Validation Triggered");
             if (validationContentDiv) {
                validationContentDiv.innerHTML = '<p>Running model validation... This may take a while.</p>';
            }

            try {
                const response = await fetch('/run_validation/', {
                    method: 'POST',
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
                     validationContentDiv.innerHTML = `
                         <p style="color: green;">Validation completed!</p>
                         <h3>Validation Results:</h3>
                         <pre>${JSON.stringify(result.results, null, 2)}</pre>
                     `;
                 }

            } catch (error) {
                console.error('Error running validation:', error);
                 if (validationContentDiv) {
                    validationContentDiv.innerHTML = `<p style="color: red;">An error occurred during validation.</p>`;
                }
            }
        });
    }

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
