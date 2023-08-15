// Replace these values with your actual scores
var formality = 0.09;
var punctuation_freq = 0.108;
var normalized_avg_word_length = 0.293;
var diversity_ratio = 0.259;
var score = 0.593;
var normalized_complexity = 0.189;
var error_score = 0.665;

var data = {
    labels: ['Formality', 'Punctuation Frequency', 'Normalized Avg. Word Length', 'Lexical Diversity', 'Rare Word Score', 'Normalized Complexity Score', 'Error Management Score'],
    datasets: [{
        label: 'Text Features Analysis',
        backgroundColor: 'rgba(75, 192, 192, 0.4)',
        borderColor: 'rgba(75, 192, 192, 1)',
        pointBackgroundColor: 'rgba(75, 192, 192, 1)',
        data: [formality, punctuation_freq, normalized_avg_word_length, diversity_ratio, score, normalized_complexity, error_score]
    }]
};

var ctx = document.getElementById('radarChart').getContext('2d');
var radarChart = new Chart(ctx, {
    type: 'radar',
    data: data,
    options: {
        scale: {
            ticks: {
                beginAtZero: true,
                max: 1
            }
        }
    }
});

