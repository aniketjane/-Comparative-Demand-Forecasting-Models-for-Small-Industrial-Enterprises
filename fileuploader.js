

input.addEventListener("change", function (e) {
  var fileName = e.target.files[0].name;
  let filedata = `
    <h4>${fileName}</h4>
    `;
  dropArea.innerHTML = filedata;
});
