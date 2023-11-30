let chatLog = '';

function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const userText = inputField.value.trim();
    inputField.value = '';
    displayMessage(userText, 'Human');

    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: userText, chat_log: chatLog })
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.response, 'AI');
        chatLog = data.chat_log;
    })
    .catch(error => console.error('Error:', error));
}

function displayMessage(message, sender) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.textContent = sender + ': ' + message;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
