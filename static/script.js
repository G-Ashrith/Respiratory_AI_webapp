const uploadBtn = document.getElementById("uploadBtn");
const audioFile = document.getElementById("audioFile");

const loading = document.getElementById("loading");

const resultArea = document.getElementById("resultArea");

const predLabel = document.getElementById("predLabel");

const explainPanel = document.getElementById("explainPanel");

let chart = null;


uploadBtn.addEventListener("click", async () => {

if(!audioFile.files.length){

alert("Please upload an audio file");

return;

}

const file = audioFile.files[0];

const formData = new FormData();

formData.append("file", file);


loading.style.display = "block";

resultArea.style.display = "none";


try{

const response = await fetch("/predict",{

method:"POST",

body:formData

});

const data = await response.json();

loading.style.display = "none";


if(data.error){

alert(data.error);

return;

}


predLabel.innerText = data.pred_label;

resultArea.style.display = "block";


/* ---------------- Probability Chart ---------------- */

const labels = Object.keys(data.probs);

const values = labels.map(x => data.probs[x]);


const ctx = document.getElementById("probChart");


if(chart) chart.destroy();


chart = new Chart(ctx,{

type:"bar",

data:{

labels:labels,

datasets:[{

label:"Prediction Probability",

data:values,

backgroundColor:"rgba(54, 162, 235, 0.6)"

}]

},

options:{

scales:{

y:{

beginAtZero:true,

max:1

}

}

}

});


/* ---------------- Explainability Panel ---------------- */

explainPanel.src = data.explainability_panel;


}catch(err){

loading.style.display = "none";

alert("Prediction failed");

console.error(err);

}

});