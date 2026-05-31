/**
 * Centralized shared Chart.js plotting library.
 * Defines reusable graphing functions with premium aesthetics.
 */

// Harmonized sleek color palette
const chartColors = {
    primary: 'rgba(59, 130, 246, 0.85)',
    primaryBorder: '#3b82f6',
    success: 'rgba(16, 185, 129, 0.85)',
    successBorder: '#10b981',
    danger: 'rgba(239, 68, 68, 0.85)',
    dangerBorder: '#ef4444',
    warning: 'rgba(245, 158, 11, 0.85)',
    warningBorder: '#f59e0b',
    info: 'rgba(6, 182, 212, 0.85)',
    infoBorder: '#06b6d4',
    slate: 'rgba(71, 85, 105, 0.85)',
    slateBorder: '#475569',
    lightGray: '#f1f5f9'
};

/**
 * Creates a premium Doughnut/Pie distribution chart.
 */
function createPieChart(canvasId, labels, values) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [chartColors.success, chartColors.danger],
                borderColor: [chartColors.successBorder, chartColors.dangerBorder],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 15,
                        padding: 20,
                        font: { size: 12, weight: 'semibold' }
                    }
                }
            },
            cutout: '70%',
            animation: {
                animateRotate: true,
                animateScale: true
            }
        }
    });
}

/**
 * Creates a premium horizontal Bar chart for Feature Importances.
 */
function createHorizontalBarChart(canvasId, labels, values) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Score',
                data: values,
                backgroundColor: chartColors.info,
                borderColor: chartColors.infoBorder,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { precision: 2 }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Creates a comparative vertical Bar chart.
 */
function createModelComparisonChart(canvasId, labels, values, isPercentage = false) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const colorTheme = isPercentage ? chartColors.slate : chartColors.primary;
    const borderTheme = isPercentage ? chartColors.slateBorder : chartColors.primaryBorder;
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colorTheme,
                borderColor: borderTheme,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    grid: { color: '#e2e8f0' },
                    min: 0,
                    max: isPercentage ? 100 : 1.0,
                    ticks: {
                        callback: function(value) {
                            return isPercentage ? value + '%' : value;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Renders a specialized Confusion Matrix bar/bubble histogram.
 */
function renderConfusionMatrix(canvasId, matrix) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const tn = matrix[0][0];
    const fp = matrix[0][1];
    const fn = matrix[1][0];
    const tp = matrix[1][1];
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Neg (Safe)', 'False Pos (Alarm)', 'False Neg (Leak)', 'True Pos (Failure)'],
            datasets: [{
                data: [tn, fp, fn, tp],
                backgroundColor: [chartColors.success, chartColors.warning, chartColors.danger, chartColors.primary],
                borderColor: [chartColors.successBorder, chartColors.warningBorder, chartColors.dangerBorder, chartColors.primaryBorder],
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}
