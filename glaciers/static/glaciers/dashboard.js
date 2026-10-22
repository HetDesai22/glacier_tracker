
document.addEventListener("DOMContentLoaded", function () {
    console.log("ClimateCraft dashboard initialized.");

    
    fetch("/api/glaciers/?format=json")
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch glacier data.");
            }
            return response.json();
        })
        .then(data => {
            console.log("Glacier data received:", data);

            
            const glacier = data[0];

            
            fetch(`/api/glaciers/${glacier.id}/areas/?format=json`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error("Failed to fetch annual area data.");
                    }
                    return response.json();
                })
                .then(areas => {
                    console.log("Annual area data received:", areas);

                    
                    const chartCanvas = document.getElementById("area-chart");

                    if (chartCanvas && areas.length > 0) {
                        if (window.glacierAreaChart) {
                            window.glacierAreaChart.destroy();
                        }

                        window.glacierAreaChart = new Chart(chartCanvas, {
                            type: "line",
                            data: {
                                labels: areas.map(area => area.year),
                                datasets: [{
                                    label: "Glacier Area (km²)",
                                    data: areas.map(area => Number(area.area_km2)),
                                    borderColor: "#2563eb",
                                    backgroundColor: "rgba(37, 99, 235, 0.12)",
                                    fill: true,
                                    tension: 0.25,
                                    pointRadius: 5,
                                    pointHoverRadius: 7
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        display: true,
                                        position: "top"
                                    },
                                    tooltip: {
                                        callbacks: {
                                            label: context =>
                                                `Area: ${context.parsed.y.toFixed(4)} km²`
                                        }
                                    }
                                },
                                scales: {
                                    x: {
                                        title: {
                                            display: true,
                                            text: "Year"
                                        }
                                    },
                                    y: {
                                        beginAtZero: false,
                                        title: {
                                            display: true,
                                            text: "Glacier Area (km²)"
                                        }
                                    }
                                }
                            }
                        });
                    }


                    
                    const results = document.getElementById("area-results");
                    const status = document.getElementById("area-status");

                    if (areas.length === 0) {
                        status.textContent = "No annual area measurements found.";
                    } else {
                        status.textContent = `${areas.length} annual records loaded.`;

                        const table = document.createElement("table");

                        table.innerHTML = `
                            <thead>
                                <tr>
                                    <th>Year</th>
                                    <th>Glacier Area (km²)</th>
                                    <th>Glacier Pixels</th>
                                    <th>Validation Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${areas.map(area => `
                                    <tr>
                                        <td>${area.year}</td>
                                        <td>${Number(area.area_km2).toFixed(4)}</td>
                                        <td>${area.glacier_pixels.toLocaleString()}</td>
                                        <td>${area.validation_status}</td>
                                    </tr>
                                `).join("")}
                            </tbody>
                        `;

                        const container = document.createElement("div");
                        container.className = "table-container";
                        container.appendChild(table);
                        results.replaceChildren(container);
                    }

                })
                .catch(error => {
                    console.error("Area API error:", error);
                    document.getElementById("area-status").textContent =
                        "Unable to load annual area measurements.";
                });


            if (glacier) {
                const summary = document.getElementById("glacier-summary");

                summary.innerHTML = `
                    <h2>${glacier.name}</h2>
                    <p>GLIMS ID: ${glacier.glims_id}</p>
                    <p>Latitude: ${glacier.latitude}</p>
                    <p>Longitude: ${glacier.longitude}</p>
                    <p>Reference Area: ${
                        glacier.reference_area_km2 ?? "Not available"
                    } km²</p>
                `;
            }

        })
        .catch(error => {
            console.error("API error:", error);
        });

});
