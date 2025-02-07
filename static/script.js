document.addEventListener('DOMContentLoaded', () => {
    console.log("script.js loaded"); // Temporary log to confirm loading
    
    // Get a reference to the Confirm button
    const confirmBtn = document.getElementById('confirmBtn');
  
    if (!confirmBtn) {
      console.error("Confirm button not found!");
      return;
    }
  
    confirmBtn.addEventListener('click', function() {
      // Get the selected stock value from the dropdown
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
  
      // Make a POST request to the /api/stockdata endpoint
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
        alert(data.message || data.error); // Display feedback
      })
      .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching data.');
      })
      .finally(() => {
        // Re-enable the button and reset its text and style
        confirmBtn.disabled = false;
        confirmBtn.innerText = "Confirm";
        confirmBtn.style.backgroundColor = "";
      });
    });
  });
  