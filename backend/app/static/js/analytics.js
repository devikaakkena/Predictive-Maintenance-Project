/**
 * Analytics page client script.
 * Asynchronously fetches and renders model accuracy, F1 comparisons, and confusion matrices.
 */

document.addEventListener("DOMContentLoaded", function() {
    console.log("Analytics page JS loaded. Initiating comparison API requests...");
    
    // Fetch multi-model statistics and confusion matrix
    fetch('/api/model-comparison')
        .then(response => {
            if (!response.ok) throw new Error("API response error");
            return response.json();
        })
        .then(data => {
            const models = Object.keys(data.metrics);
            const accuracies = models.map(m => data.metrics[m].accuracy * 100);
            const f1Scores = models.map(m => data.metrics[m].f1_score);
            
            // 1. Draw comparative classifiers Accuracy bar chart
            createModelComparisonChart("modelAccuracyChart", models, accuracies, true);
            
            // 2. Draw comparative classifiers F1-Score bar chart
            createModelComparisonChart("modelF1Chart", models, f1Scores, false);
            
            // 3. Draw true vs predicted Confusion Matrix histogram
            renderConfusionMatrix("confusionMatrixChart", data.confusion_matrix);
        })
        .catch(err => console.error("Error drawing model comparison charts: ", err));
});
