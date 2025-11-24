// Helper to sync slider with value display
function setupSlider(id, valId) {
    const slider = document.getElementById(id);
    const display = document.getElementById(valId);
    slider.addEventListener('input', () => {
        display.innerText = slider.value;
    });
}

// Setup all sliders
['visual', 'audio', 'source', 'logic', 'va', 'vc', 'ac'].forEach(key => setupSlider(`score-${key}`, `val-${key}`));
setupSlider('score-final', 'val-final');

// Parse query params
const params = new URLSearchParams(window.location.search);
const link = params.get('link') || '';
const caption = params.get('caption') || '';

document.getElementById('input-link').value = link;
document.getElementById('input-caption').value = caption;

// Handle Submit
document.getElementById('label-form').addEventListener('submit', (e) => {
    e.preventDefault();
    
    const submitBtn = e.target.querySelector('button');
    submitBtn.innerText = 'Saving...';
    submitBtn.disabled = true;

    const payload = {
        link: document.getElementById('input-link').value,
        caption: document.getElementById('input-caption').value,
        labels: {
            visual_integrity_score: parseInt(document.getElementById('score-visual').value),
            audio_integrity_score: parseInt(document.getElementById('score-audio').value),
            source_credibility_score: parseInt(document.getElementById('score-source').value),
            logical_consistency_score: parseInt(document.getElementById('score-logic').value),
            emotional_manipulation_score: 5, // Default or add another slider
            
            video_audio_score: parseInt(document.getElementById('score-va').value),
            video_caption_score: parseInt(document.getElementById('score-vc').value),
            audio_caption_score: parseInt(document.getElementById('score-ac').value),
            
            final_veracity_score: parseInt(document.getElementById('score-final').value),
            reasoning: document.getElementById('input-reasoning').value
        }
    };

    chrome.runtime.sendMessage({type: 'SAVE_MANUAL', payload: payload}, (response) => {
        if (response && response.success) {
            alert('Label Saved Successfully!');
            window.close();
        } else {
            alert('Error saving label: ' + (response ? response.error : 'Unknown'));
            submitBtn.innerText = 'Save Label';
            submitBtn.disabled = false;
        }
    });
});