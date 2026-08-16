// Basic client-side validation before the form is submitted to Flask.
// Server-side validation in app.py is the authoritative check;
// this just gives the user immediate feedback.

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", function (event) {
        const absences = document.getElementById("absences");
        const failures = document.getElementById("failures");
        const g1 = document.getElementById("g1");
        const g2 = document.getElementById("g2");

        const errors = [];

        if (absences.value === "" || absences.value < 0 || absences.value > 100) {
            errors.push("Absences must be between 0 and 100.");
        }
        if (failures.value === "" || failures.value < 0 || failures.value > 4) {
            errors.push("Past class failures must be between 0 and 4.");
        }
        if (g1.value === "" || g1.value < 0 || g1.value > 20) {
            errors.push("Previous marks (G1) must be between 0 and 20.");
        }
        if (g2.value === "" || g2.value < 0 || g2.value > 20) {
            errors.push("Internal marks (G2) must be between 0 and 20.");
        }

        if (errors.length > 0) {
            event.preventDefault();
            alert(errors.join("\n"));
        }
    });
});
