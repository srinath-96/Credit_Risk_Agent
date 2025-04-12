import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'; // Import the CSS

// --- Configuration ---
// Use environment variable for API URL (Vite example)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';
// For Create React App, use process.env.REACT_APP_API_URL
console.log("API URL:", API_URL); // Verify the URL is read correctly


function App() {
  // --- State Variables ---
  const [messages, setMessages] = useState([
    // Initial welcome message
    { id: Date.now(), role: 'assistant', content: 'Hello! How can I help you with credit risk analysis today?', plot_urls: [] }
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [datasetFile, setDatasetFile] = useState(null);

  // --- Refs ---
  const chatEndRef = useRef(null); // Ref to scroll to bottom

  // --- Effects ---
  // Scroll to bottom whenever messages update
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- API Call Function ---
  const sendMessageToAPI = useCallback(async (messageToSend, historyToSend) => {
    setIsLoading(true);
    setError(null);
    console.log("Sending to API:", { message: messageToSend, history: historyToSend });

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: messageToSend,
            history: historyToSend
        }),
      });

      // Check for network/server errors first
      if (!response.ok) {
        let errorMsg = `Server Error: ${response.status} ${response.statusText}`;
        try {
          // Try to get more specific error from backend JSON response
          const errData = await response.json();
          errorMsg = errData.error || errorMsg;
        } catch (e) {
            console.warn("Could not parse error response JSON:", e);
            // If JSON parsing fails, use the raw text if possible
            try {
                const textError = await response.text();
                if (textError) errorMsg = `${errorMsg} - ${textError}`;
            } catch(textErr) { /* Ignore */}
        }
        throw new Error(errorMsg);
      }

      // Parse the successful JSON response
      const data = await response.json();
      console.log("Received from API:", data);

      // Add assistant response to history state
      const assistantMessage = {
        id: Date.now(), // Unique key for React list rendering
        role: 'assistant',
        content: data.response || "Sorry, I received an empty response.", // Provide fallback content
        plot_urls: data.plot_urls || [] // Ensure plot_urls is always an array
      };
      // Use functional update to ensure we have the latest state
      setMessages(prevMessages => [...prevMessages, assistantMessage]);

    } catch (err) {
      console.error("API Communication Error:", err);
      setError(err.message || "Failed to connect to the server or process the request.");
      // Optionally add an error message bubble to the chat
      // setMessages(prevMessages => [...prevMessages, { id: Date.now(), role: 'assistant', content: `Error: ${err.message}`, plot_urls: [] }]);
    } finally {
      setIsLoading(false); // Reset loading state regardless of success/failure
    }
  }, []); // useCallback with empty dependency array as it doesn't depend on component state directly

  // --- File Upload Handler ---
  const handleFileChange = (event) => {
    setDatasetFile(event.target.files[0]);
  };

  const uploadDataset = async () => {
    if (!datasetFile) return;

    const formData = new FormData();
    formData.append('dataset', datasetFile);

    try {
      const response = await fetch(`${API_URL}/api/upload-dataset`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload dataset');
      }

      const result = await response.json();
      console.log('Dataset uploaded successfully:', result);
      // Optionally, you can send a message to the chat indicating success
      setMessages(prevMessages => [...prevMessages, { id: Date.now(), role: 'assistant', content: 'Dataset uploaded successfully!', plot_urls: [] }]);

    } catch (error) {
      console.error('Error uploading dataset:', error);
      setError(error.message || "Failed to upload dataset.");
    }
  };

  // --- Event Handlers ---
  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSendMessage = () => {
    const trimmedInput = userInput.trim();
    if (!trimmedInput || isLoading) {
      return; // Prevent sending empty or duplicate messages while loading
    }

    // Add user message immediately to the UI state
    const newUserMessage = {
        id: Date.now(), // Unique key
        role: 'user',
        content: trimmedInput,
        plot_urls: []
    };
    // Update messages state *before* sending API call
    const currentMessages = [...messages, newUserMessage];
    setMessages(currentMessages);

    // Clear input field *after* updating state
    setUserInput('');

    // Prepare history to send (exclude the new user message we just added)
    const historyToSend = messages;

    // Call the API function
    sendMessageToAPI(trimmedInput, historyToSend);
  };


  // Handle Enter key press in input field
  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Prevent default form submission/newline
      handleSendMessage();
    }
  };

  // --- Rendering ---
  return (
    <div className="app-container">
      <header>
        Credit Risk Assistant
      </header>
  
      {/* File Upload Input */}
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={uploadDataset}>Upload Dataset</button>
  
      <div className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-bubble ${msg.role}`}>
            {typeof msg.content === 'string' ? msg.content.split('\n').map((line, i) => (
              <span key={i}>{line}<br /></span>
            )) : 'Invalid message content'}
  
            {Array.isArray(msg.plot_urls) && msg.plot_urls.length > 0 && (
              <div className="plot-container">
                {msg.plot_urls.map((url, plotIndex) => (
                  <img
                    key={plotIndex}
                    src={`${API_URL}${url}`}
                    alt={`Generated plot ${plotIndex + 1}`}
                    onError={(e) => {
                      console.warn(`Failed to load image: ${API_URL}${url}`);
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} style={{ height: '1px' }} />
      </div>
  
      <div className="status-indicator">
        {isLoading && <div className="loading-indicator">Assistant is thinking...</div>}
        {error && <div className="error-message">{error}</div>}
      </div>
  
      <div className="input-area">
        <input
          type="text"
          value={userInput}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isLoading}
          aria-label="Chat input"
        />
        <button onClick={handleSendMessage} disabled={isLoading || !userInput.trim()} aria-label="Send message">
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
