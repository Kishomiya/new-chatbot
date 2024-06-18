function toggleChatbot() {
    var chatbot = document.getElementById("chatbot");
    if (chatbot.style.display === "none" || chatbot.style.display === "") {
        chatbot.style.display = "block";
    } else {
        chatbot.style.display = "none";
    }
}

function sendButton() {
    var userText = document.getElementById("textInput").value;
    if (userText.trim() === "") {
        return;
    }
    
    // Append user message to chatbox
    var userHtml = '<div class="msg right-msg"><div class="msg-bubble">' + userText + '</div></div>';
    document.getElementById("chatbox").innerHTML += userHtml;

    // Clear text input
    document.getElementById("textInput").value = "";

    // Send the message to the server
    fetch('/get', {
        method: 'POST',
        body: JSON.stringify({ message: userText }),
        headers: { 'Content-Type': 'application/json' }
    }).then(response => response.json())
    .then(data => {
        // Append bot response to chatbox
        var botHtml = '<div class="msg left-msg"><div class="msg-bubble">' + data.reply + '</div></div>';
        document.getElementById("chatbox").innerHTML += botHtml;

        // Scroll to the bottom of the chatbox
        var chatbox = document.getElementById("chatbox");
        chatbox.scrollTop = chatbox.scrollHeight;
    }).catch(error => {
        console.error('Error:', error);
    });
}
