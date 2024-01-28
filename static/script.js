function predecir() {
    var edad = document.getElementById("edad").value;
    var sexo = document.getElementById("sexo").value;

    fetch('http://127.0.0.1:5000/predecir', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'edad=' + edad + '&sexo=' + sexo,
    })
    .then(response => response.json())
    .then(resultado => {
        alert("Nivel de estrés: " + resultado.Nivel_de_estres);

        document.getElementById("resultado-container").innerHTML = "Nivel de estrés: " + resultado.Nivel_de_estres;
    });
}
