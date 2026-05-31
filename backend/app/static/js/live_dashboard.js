/**
 * Industrial Sensor Streaming Telemetry client controller.
 * Handles AJAX streaming polling and dynamic UI status badge styling.
 */

document.addEventListener("DOMContentLoaded", function() {
    console.log("Telemetry simulation client controller loaded. Awaiting start command...");
    
    let isStreaming = false;
    let streamInterval = null;
    
    // UI Selectors
    const toggleSimBtn = document.getElementById("toggleSimBtn");
    const liveModeBadge = document.getElementById("liveModeBadge");
    
    const liveAirTemp = document.getElementById("liveAirTemp");
    const liveProcTemp = document.getElementById("liveProcTemp");
    const liveSpeed = document.getElementById("liveSpeed");
    const liveTorque = document.getElementById("liveTorque");
    const liveToolWear = document.getElementById("liveToolWear");
    
    const liveStatusBadge = document.getElementById("liveStatusBadge");
    const liveAlertBox = document.getElementById("liveAlertBox");
    
    /**
     * Polls the backend simulation stream API and updates UI nodes.
     */
    function fetchTelemetryData() {
        fetch('/api/simulation-stream')
            .then(response => {
                if (!response.ok) throw new Error("Telemetry API error");
                return response.json();
            })
            .then(data => {
                // 1. Update text values
                liveAirTemp.textContent = data.air_temp + " K";
                liveProcTemp.textContent = data.process_temp + " K";
                liveSpeed.textContent = data.speed + " rpm";
                liveTorque.textContent = data.torque + " Nm";
                liveToolWear.textContent = data.tool_wear + " min";
                
                // 2. Clear inline standby styles
                liveStatusBadge.removeAttribute("style");
                
                // 3. Update dynamic status classes
                liveStatusBadge.className = "fs-5 fw-bold px-4 py-2 rounded-pill d-inline-block border shadow-sm";
                
                if (data.status === "Healthy") {
                    liveStatusBadge.classList.add("bg-success", "bg-opacity-10", "text-success", "border-success");
                    liveStatusBadge.innerHTML = '<i class="fa-solid fa-circle-check me-2"></i>HEALTHY';
                    
                    liveAlertBox.className = "alert alert-success py-2 px-3 m-0 d-inline-block shadow-sm";
                    liveAlertBox.innerHTML = '<i class="fa-solid fa-circle-check text-success me-2"></i>Anomalies Check: OK. Telemetry stable.';
                } else if (data.status === "Warning") {
                    liveStatusBadge.classList.add("bg-warning", "bg-opacity-10", "text-warning", "border-warning");
                    liveStatusBadge.innerHTML = '<i class="fa-solid fa-circle-exclamation me-2"></i>WARNING';
                    
                    liveAlertBox.className = "alert alert-warning py-2 px-3 m-0 d-inline-block shadow-sm";
                    liveAlertBox.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-warning me-2"></i>Warning: High friction torque / tool wear limit!';
                } else if (data.status === "Critical") {
                    liveStatusBadge.classList.add("bg-danger", "bg-opacity-10", "text-danger", "border-danger");
                    liveStatusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark me-2"></i>CRITICAL';
                    
                    liveAlertBox.className = "alert alert-danger py-2 px-3 m-0 d-inline-block shadow-sm";
                    liveAlertBox.innerHTML = '<i class="fa-solid fa-skull-crossbones text-danger me-2"></i>Failure Caught: Friction anomaly detected by Random Forest model!';
                }
            })
            .catch(err => {
                console.error("Telemetry streaming error: ", err);
                liveAlertBox.className = "alert alert-danger py-2 px-3 m-0 d-inline-block shadow-sm";
                liveAlertBox.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-danger me-2"></i>Error streaming telemetry. Retrying...';
            });
    }
    
    // Toggle Stream button logic
    toggleSimBtn.addEventListener("click", function() {
        if (isStreaming) {
            // Stop streaming
            isStreaming = false;
            clearInterval(streamInterval);
            
            toggleSimBtn.className = "btn btn-primary btn-sm rounded shadow-sm px-3 py-1.5 fw-semibold";
            toggleSimBtn.innerHTML = '<i class="fa-solid fa-play me-1"></i>Start Simulation Stream';
            
            liveModeBadge.className = "badge bg-secondary py-1.5 px-3 rounded-pill text-uppercase";
            liveModeBadge.textContent = "Offline";
            
            liveStatusBadge.className = "fs-5 fw-bold px-4 py-2 rounded-pill d-inline-block border shadow-sm";
            liveStatusBadge.style.backgroundColor = "#e2e8f0";
            liveStatusBadge.style.color = "#475569";
            liveStatusBadge.innerHTML = '<i class="fa-solid fa-power-off me-2"></i>STANDBY';
            
            liveAlertBox.className = "alert alert-secondary py-2 px-3 m-0 d-inline-block shadow-sm";
            liveAlertBox.innerHTML = '<i class="fa-solid fa-circle-info text-primary me-2"></i>Telemetry stream currently paused.';
        } else {
            // Start streaming
            isStreaming = true;
            fetchTelemetryData(); // Execute once immediately
            streamInterval = setInterval(fetchTelemetryData, 3000); // Poll every 3 seconds
            
            toggleSimBtn.className = "btn btn-danger btn-sm rounded shadow-sm px-3 py-1.5 fw-semibold";
            toggleSimBtn.innerHTML = '<i class="fa-solid fa-pause me-1"></i>Pause Simulation Stream';
            
            liveModeBadge.className = "badge bg-success py-1.5 px-3 rounded-pill text-uppercase text-white";
            liveModeBadge.textContent = "Streaming";
        }
    });
});
