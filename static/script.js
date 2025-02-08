document.addEventListener('DOMContentLoaded', () => {
  console.log("script.js loaded"); // For debugging
  
  // --- Confirm Button: Trigger stock data download and preprocessing ---
  const confirmBtn = document.getElementById('confirmBtn');
  if (!confirmBtn) {
    console.error("Confirm button not found!");
  } else {
    confirmBtn.addEventListener('click', function() {
      const stockSelect = document.getElementById('stockSelect');
      const selectedStock = stockSelect.value;
  
      if (!selectedStock) {
        alert("Please select a stock.");
        return;
      }
  
      // Provide immediate visual feedback:
      confirmBtn.disabled = true;
      confirmBtn.innerText = "Loading...";
      confirmBtn.style.backgroundColor = "#5cb85c"; // A green shade
  
      // Make the API call to trigger stock data download and preprocessing
      fetch('/api/stockdata', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ stock: selectedStock })
      })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Log the API response
        alert(data.message || data.error); // Show a confirmation or error message
      })
      .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching data.');
      })
      .finally(() => {
        // Reset the button
        confirmBtn.disabled = false;
        confirmBtn.innerText = "Confirm";
        confirmBtn.style.backgroundColor = "";
      });
    });
  }
  
  // --- Predict Button: Trigger prediction using cached data and the GRU model ---
  // We handle form submission for the Predict button.
  const stockForm = document.getElementById('stockForm');
  if (!stockForm) {
    console.error("Stock form not found!");
    return;
  }
  
  stockForm.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission
    
    const stockSelect = document.getElementById('stockSelect');
    const selectedStock = stockSelect.value;
  
    if (!selectedStock) {
      alert("Please select a stock.");
      return;
    }
    
    // Provide immediate visual feedback for prediction
    const predictBtn = stockForm.querySelector('button[type="submit"]');
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.innerText = "Predicting...";
      predictBtn.style.backgroundColor = "#5cb85c"; // A green shade
    }
    
    // Make the API call to trigger prediction
    fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ stock: selectedStock })
    })
    .then(response => response.json())
    .then(data => {
      console.log(data); // Log the API response
      if (data.predictions) {
        alert("Predictions for next 2 days: " + data.predictions.join(", "));
      } else {
        alert(data.message || data.error);
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred during prediction.');
    })
    .finally(() => {
      if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.innerText = "Predict";
        predictBtn.style.backgroundColor = "";
      }
    });
  });
});
