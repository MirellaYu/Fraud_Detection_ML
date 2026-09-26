document.getElementById("btnPredict").addEventListener("click", async function () {

    const payload = {

        type: document.getElementById("tipo").value,
        amount: parseFloat(
            document.getElementById("monto").value
        ),
        oldbalanceOrg: parseFloat(
            document.getElementById("saldo_ant_origen").value
        ),
        newbalanceOrig: parseFloat(
            document.getElementById("saldo_nuevo_origen").value
        ),
        oldbalanceDest: parseFloat(
            document.getElementById("saldo_ant_destino").value
        ),
        newbalanceDest: parseFloat(
            document.getElementById("saldo_nuevo_destino").value
        )
    };


    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(payload)
    });

    const result = await response.json();
    console.log("Respuesta del servidor:", result);

    const resultBox = document.getElementById("resultBox");
    resultBox.style.display = "flex";
    const resultText = resultBox.querySelector("span");
    const resultScore = resultBox.querySelector(".result-score");

    const probability = (result.probability * 100).toFixed(1);

    if (result.prediction === 1) {

        resultBox.classList.remove("safe");
        resultBox.classList.add("risk");

        resultText.textContent = "Riesgo de fraude";
        resultScore.textContent = `Probabilidad estimada: ${probability}%`;
    } else {
        resultBox.classList.remove("risk");
        resultBox.classList.add("safe");
        resultText.textContent = "Transacción no fraudulenta";
        resultScore.textContent = `Probabilidad estimada: ${probability}%`;
    }
});