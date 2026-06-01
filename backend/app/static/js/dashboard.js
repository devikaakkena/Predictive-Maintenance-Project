/**
 * Operations Dashboard client script.
 * Asynchronously fetches and triggers dashboard main page graphs.
 */

document.addEventListener("DOMContentLoaded", function() {
    console.log("Operations Dashboard JS loaded. Initiating Chart API requests...");
    
    // 1. Fetch and render Failure Distribution Doughnut Chart
    fetch('/api/prediction-summary')
        .then(response => {
            if (!response.ok) throw new Error("API response error");
            return response.json();
        })
        .then(data => {
            window.failurePieChart = createPieChart("failureDistributionChart", data.labels, data.values);
        })
        .catch(err => console.error("Error drawing prediction summary chart: ", err));

    // 2. Fetch and render Optimal Feature Importances horizontal bar chart
    fetch('/api/feature-importance')
        .then(response => {
            if (!response.ok) throw new Error("API response error");
            return response.json();
        })
        .then(data => {
            createHorizontalBarChart("featureImportanceChart", data.features, data.importances);
        })
        .catch(err => console.error("Error drawing feature importance chart: ", err));
});
