async function loadData() {
    try {
        const response = await fetch("../output/output.json"); 
        const data = await response.json();

        document.getElementById("sentiment").textContent =
            JSON.stringify(data.sentiment_metrics, null, 4);

        document.getElementById("topics").textContent =
            JSON.stringify(data.topics, null, 4);

        document.getElementById("insights").textContent =
            JSON.stringify(data.insights, null, 4);

    } catch (error) {
        console.error("Error loading JSON:", error);
    }
}

loadData();
