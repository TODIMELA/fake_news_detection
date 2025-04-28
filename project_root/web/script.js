document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('newsForm');
    const loadingAnimation = document.getElementById('loading');
    const resultContainer = document.getElementById('resultContainer');
    const animationContainer = document.getElementById('animationContainer');
    const resultText = document.getElementById('resultText');
    const explanationContainer = document.getElementById('explanation');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        loadingAnimation.style.display = 'block';
        resultContainer.style.display = 'none';

        const formData = new FormData(form);
        const data = {
            input_type: formData.get('inputType'),
            input_value: formData.get('inputValue')
        };

        fetch('/api/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            loadingAnimation.style.display = 'none';
            resultContainer.style.display = 'block';

            if (result.classification === 'Real') {
                resultText.textContent = 'Real News';
                resultText.style.color = 'green';
                animationContainer.innerHTML = '<div class="checkmark"></div>';
            } else {
                resultText.textContent = 'Fake News';
                resultText.style.color = 'red';
                animationContainer.innerHTML = '<div class="cross"></div>';
            }

            if (result.explanation) {
                explanationContainer.innerHTML = '<h3>Explanation:</h3><p>' + result.explanation + '</p>';
                const explanationText = explanationContainer.querySelector('p');
                  result.important_words.forEach(word => {
                    const regex = new RegExp(`\\b(${word})\\b`, 'gi');
                    explanationText.innerHTML = explanationText.innerHTML.replace(regex, `<span style="color: ${result.classification === 'Real' ? 'green' : 'red'}; font-weight: bold;">$1</span>`);
                });
            }

            
        })
        .catch(error => {
            console.error('Error:', error);
            loadingAnimation.style.display = 'none';
            resultText.textContent = 'Error processing request.';
            resultText.style.color = 'black';
            resultContainer.style.display = 'block';
            animationContainer.innerHTML = ''; 
        });
    });
});