function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    if (!fileInput.files[0]) {
        alert("Please upload an image.");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("resultBox").style.display = "block";

        // Display processed image
        document.getElementById("processedImg").src = data.processed_image;

        // Display predictions
        let list = document.getElementById("predictionList");
        list.innerHTML = "";

        data.predictions.forEach(pred => {
            let item = document.createElement("li");
            item.innerHTML = `${pred.rank}. ${pred.label}`;
            list.appendChild(item);
        });
    })
    .catch(err => {
        alert("Error occurred: " + err);
    });
}
// Aesthetic uploader + drag-drop + preview + POST to /predict
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const processedImg = document.getElementById('processedImg');
const originalImg = document.getElementById('originalImg');
const resultArea = document.getElementById('resultArea');
const predictionRow = document.getElementById('predictionRow');
const toast = document.getElementById('toast');

let selectedFile = null;

// helper: show toast
function showToast(msg, timeout = 2800) {
  toast.hidden = false;
  toast.textContent = msg;
  setTimeout(() => toast.hidden = true, timeout);
}

// open file dialog when clicking "browse"
browseBtn.addEventListener('click', (e) => {
  e.preventDefault();
  fileInput.click();
});

// file selected through dialog
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

// drag events
['dragenter','dragover'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});
['dragleave','dragend','drop'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

// drop -> get file
dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (!dt) return;
  const file = dt.files[0];
  if (file) handleFile(file);
});

// allow keyboard enter on drop zone
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

// clear selection
clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = "";
  resultArea.hidden = true;
  processedImg.src = "";
  originalImg.src = "";
  predictionRow.innerHTML = "";
  predictBtn.disabled = true;
  showToast("Cleared");
});

// handle a file object (validate + preview)
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast("Please upload an image file (jpg/png).");
    return;
  }
  selectedFile = file;
  predictBtn.disabled = false;

  // show original preview (object URL)
  originalImg.src = URL.createObjectURL(file);

  // show result area (processed image replaced once server returns)
  resultArea.hidden = false;
  processedImg.src = '/static/placeholder.png'; // optional placeholder or keep blank
  predictionRow.innerHTML = "";
  showToast("Image ready. Click Predict.");
}

// Predict click -> upload to server
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) { showToast("No image selected"); return; }

  predictBtn.disabled = true;
  predictBtn.textContent = "Working...";
  predictionRow.innerHTML = "";

  try {
    const form = new FormData();
    form.append('image', selectedFile);

    const res = await fetch('/predict', { method: 'POST', body: form });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || 'Server error');
    }
    const data = await res.json();

    // show processed image
    if (data.processed_image) processedImg.src = data.processed_image;

    // render predictions horizontally (no percentages)
    predictionRow.innerHTML = '';
    if (Array.isArray(data.predictions) && data.predictions.length) {
      data.predictions.forEach(p => {
        const d = document.createElement('div');
        d.className = 'pred-item';
        d.textContent = `${p.rank}. ${p.label}`;
        predictionRow.appendChild(d);
      });
    } else {
      predictionRow.textContent = 'No predictions';
    }

    showToast("Prediction complete");
  } catch (err) {
    console.error(err);
    showToast('Prediction failed. See console');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
// -----------------------------
// Topbar show/hide on scroll
// -----------------------------
(function () {
  const topbar = document.getElementById('topbar');
  let lastScroll = window.scrollY || 0;
  let ticking = false;

  function onScroll() {
    const current = window.scrollY || 0;

    // show topbar when scrolled more than 120px down
    if (current > 120) {
      topbar.classList.add('visible');
      topbar.setAttribute('aria-hidden', 'false');
    } else {
      topbar.classList.remove('visible');
      topbar.setAttribute('aria-hidden', 'true');
    }

    lastScroll = current;
    ticking = false;
  }

  window.addEventListener('scroll', () => {
    if (!ticking) {
      window.requestAnimationFrame(onScroll);
      ticking = true;
    }
  }, { passive: true });

  // also check on load (in case the page opened lower)
  document.addEventListener('DOMContentLoaded', onScroll);
})();
