document.addEventListener('DOMContentLoaded', () => {
    // General setup for navigation
    const navLinks = document.querySelectorAll('nav a');
    const sections = document.querySelectorAll('main section');

    const showSection = (sectionId) => {
        sections.forEach(section => {
            section.classList.toggle('hidden', section.id !== sectionId);
        });
    };

    navLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const sectionId = link.getAttribute('href').substring(1);
            showSection(sectionId);
            if (sectionId === 'model-info') fetchModelInfo();
            if (sectionId === 'validation') fetchValidation();
        });
    });

    showSection('prediction');

    const predictionResultDiv = document.getElementById('prediction-result');
    const modelInfoContentDiv = document.getElementById('model-info-content');
    const validationContentDiv = document.getElementById('validation-content');
    const retrainingResultDiv = document.getElementById('retraining-result');

    // Amino acid color scheme
    const residueColors = {
        'ALA': 'cyan',
        'ARG': 'blue',
        'ASN': 'green',
        'ASP': 'lime',
        'CYS': 'yellow',
        'GLN': 'magenta',
        'GLU': 'orange',
        'GLY': 'red',
        'HIS': 'purple',
        'ILE': 'teal',
        'LEU': 'olive',
        'LYS': 'silver',
        'MET': 'gold',
        'PHE': 'maroon',
        'PRO': 'pink',
        'SER': 'brown',
        'THR': 'navy',
        'TRP': 'indigo',
        'TYR': 'coral',
        'VAL': 'white'
    };

    const renderPredictionResults = (result) => {
        predictionResultDiv.innerHTML = '';

        const successMessage = document.createElement('p');
        successMessage.style.color = 'green';
        successMessage.textContent = 'Prediction successful!';
        predictionResultDiv.appendChild(successMessage);

        const blob = new Blob([result.csv_data], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction.csv';
        a.textContent = 'Download prediction.csv';
        predictionResultDiv.appendChild(a);

        if (result.pdb_data) {
            const viewerContainer = document.createElement('div');
            viewerContainer.id = 'protein-viewer';
            viewerContainer.style.height = '500px';
            viewerContainer.style.width = '100%';
            viewerContainer.style.position = 'relative';
            viewerContainer.style.border = '1px solid #ccc';
            predictionResultDiv.appendChild(viewerContainer);

            let viewer = $3Dmol.createViewer(viewerContainer);
            viewer.addModel(result.pdb_data, "pdb");
            
            // Apply color scheme
            const atoms = viewer.getModel().selectedAtoms({});
            for (let i = 0; i < atoms.length; i++) {
                const atom = atoms[i];
                atom.color = residueColors[atom.resn] || 'gray';
            }
            
            viewer.setStyle({}, { stick: {} });
            viewer.zoomTo();
            viewer.render();
            viewer.resize();
            
            fetchAndDisplayDescription(result.pdb_data);
        }
    };
    
    const fetchAndDisplayDescription = async (pdbData) => {
        const descriptionContainer = document.createElement('div');
        descriptionContainer.id = 'protein-description';
        descriptionContainer.innerHTML = '<p>Generating description...</p>';
        predictionResultDiv.appendChild(descriptionContainer);

        try {
            const response = await fetch('/describe_protein/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pdb_data: pdbData })
            });
            const result = await response.json();
            if (!response.ok) throw result;
            descriptionContainer.innerHTML = `<h3>Protein Description:</h3><p>${result.description}</p>`;
        } catch (error) {
            handleError(descriptionContainer, error, 'Failed to generate description');
        }
    };


    const handleError = (div, error, prefix = 'Error') => {
        console.error(prefix, error);
        let errorMessage = error.detail || error.message || 'An unknown error occurred.';
        div.innerHTML = `<p style="color: red;">${prefix}: ${errorMessage}</p>`;
    };

    const predictSequenceForm = document.getElementById('predict-sequence-form');
    if (predictSequenceForm) {
        predictSequenceForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            predictionResultDiv.innerHTML = '<p>Running prediction...</p>';
            const sequence = document.getElementById('protein-sequence-input').value;
            try {
                const response = await fetch('/predict/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ sequence })
                });
                const result = await response.json();
                if (!response.ok) throw result;
                renderPredictionResults(result);
            } catch (error) {
                handleError(predictionResultDiv, error, 'Prediction failed');
            }
        });
    }

    const predictFileForm = document.getElementById('predict-file-form');
    if (predictFileForm) {
        predictFileForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            predictionResultDiv.innerHTML = '<p>Running prediction...</p>';
            const fileInput = document.getElementById('fasta-file-input');
            if (!fileInput.files.length) {
                handleError(predictionResultDiv, { message: 'Please select a file.' });
                return;
            }
            const formData = new FormData();
            formData.append('fasta_file', fileInput.files[0]);
            try {
                const response = await fetch('/predict/', { method: 'POST', body: formData });
                const result = await response.json();
                if (!response.ok) throw result;
                renderPredictionResults(result);
            } catch (error) {
                handleError(predictionResultDiv, error, 'Prediction failed');
            }
        });
    }

    const fetchModelInfo = async () => {
        modelInfoContentDiv.innerHTML = '<p>Loading...</p>';
        try {
            const response = await fetch('/model_info/');
            const result = await response.json();
            if (!response.ok) throw result;
            modelInfoContentDiv.innerHTML = `
                <h3>Model Structure:</h3><pre>${result.model_summary || 'N/A'}</pre>
                <h3>Last Validation:</h3><pre>${JSON.stringify(result.last_validation_results, null, 2) || 'N/A'}</pre>
            `;
        } catch (error) {
            handleError(modelInfoContentDiv, error, 'Failed to load model info');
        }
    };

    const fetchValidation = async () => {
        validationContentDiv.innerHTML = '<p>Loading...</p>';
        try {
            const response = await fetch('/validation/');
            const result = await response.json();
            if (!response.ok) throw result;
            validationContentDiv.innerHTML = `<h3>Latest Run:</h3><pre>${JSON.stringify(result, null, 2)}</pre>`;
        } catch (error) {
            handleError(validationContentDiv, error, 'Failed to load validation results');
        }
    };

    const runValidationButton = document.getElementById('run-validation-button');
    if (runValidationButton) {
        runValidationButton.addEventListener('click', async () => {
            validationContentDiv.innerHTML = '<p>Running validation...</p>';
            try {
                const response = await fetch('/run_validation/', { method: 'POST' });
                const result = await response.json();
                if (!response.ok) throw result;
                validationContentDiv.innerHTML = 
                    `<p style="color: green;">Validation successful!</p><h3>Results:</h3><pre>${JSON.stringify(result.results, null, 2)}</pre>`;
            } catch (error) {
                handleError(validationContentDiv, error, 'Validation failed');
            }
        });
    }

    const retrainingForm = document.getElementById('retraining-form');
    if (retrainingForm) {
        retrainingForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            retrainingResultDiv.innerHTML = '<p>Starting retraining...</p>';
            const formData = new FormData(retrainingForm);
            try {
                const response = await fetch('/retrain_model/', { method: 'POST', body: formData });
                const result = await response.json();
                if (!response.ok) throw result;
                retrainingResultDiv.innerHTML = `<p style="color: green;">Retraining successful!</p><pre>${JSON.stringify(result, null, 2)}</pre>`;
            } catch (error) {
                handleError(retrainingResultDiv, error, 'Retraining failed');
            }
        });
    }
});
